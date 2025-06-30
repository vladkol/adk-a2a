# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A2A Server that acts as a proxy in front of an ADK agent"""

import base64
import inspect
import logging
from pathlib import Path
from typing import Dict, Callable, Optional
from typing_extensions import override

import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from a2a.types import (
    AgentCard,
    Message,
    DataPart, TextPart, FileWithBytes, FileWithUri, FilePart, Part,
    Role,
    TaskState
)

from google.genai import types

from google.adk.agents import BaseAgent
from google.adk.artifacts import BaseArtifactService, InMemoryArtifactService
from google.adk.events import Event
from google.adk.sessions import BaseSessionService

from starlette.datastructures import URL
from starlette.requests import Request
from starlette.responses import JSONResponse

from vertexai.preview import reasoning_engines


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from a2a.server.tasks import TaskUpdater

class ADKExecutor(AgentExecutor):
    """Executes the ADK agent logic in response to A2A requests."""

    def __init__(
        self,
        adk_agent: BaseAgent,
        task_store: TaskStore,
        session_service_builder: Optional[Callable[..., "BaseSessionService"]] = None,
        artifact_service_builder: Optional[Callable[..., "BaseArtifactService"]] = None,
        enable_tracing: Optional[bool] = False,
        env_vars: Optional[Dict[str, str]] = None,
        transfer_function_calls: bool = False,
    ):
        self.adk_agent = adk_agent
        self.task_store = task_store
        self.transfer_function_calls = transfer_function_calls
        self.adk_app = reasoning_engines.AdkApp(
            agent=adk_agent,
            enable_tracing=enable_tracing or False,
            session_service_builder=session_service_builder,
            artifact_service_builder=artifact_service_builder,
            env_vars=env_vars
        )
        self.adk_app.set_up()
        self.in_memory_artifact_service = InMemoryArtifactService()

    @property
    def artifact_service(self) -> BaseArtifactService:
        if (
            hasattr(self.adk_app, "_tmpl_attrs")
            and "artifact_service" in self.adk_app._tmpl_attrs
        ):
            return self.adk_app._tmpl_attrs[
                "artifact_service"
            ]
        else:
            return self.in_memory_artifact_service

    def _convert_a2a_part_to_genai(
            self,
            returned_part: Part
        ) -> Optional[types.Part]:
        """Convert a single A2A Part type into a Google Gen AI Part type."""
        part = returned_part.root
        if isinstance(part, TextPart):
            return types.Part(text=part.text)
        elif isinstance(part, DataPart):
            return types.Part(text=part.model_dump_json())
        elif isinstance(part, FilePart):
            if isinstance(part.file, FileWithUri):
                return types.Part(
                    file_data=types.FileData(
                        file_uri=part.file.uri,
                        mime_type=part.file.mimeType
                    )
                )
            part_bytes = base64.decodebytes(
                part.file.bytes.encode("ascii")
            )
            if isinstance(part.file, FileWithBytes):
                return types.Part(
                    inline_data=types.Blob(
                        data=part_bytes, mime_type=part.file.mimeType
                    )
                )
            return None
        return None

    def _convert_genai_part_to_a2a(self, part: types.Part) -> Optional[Part]:
        """Convert a single Google Gen AI Part type into an A2A Part type."""
        if part.text:
            return Part(root=TextPart(text=part.text))
        if part.file_data:
            return Part(root=FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri, # type: ignore
                    mimeType=part.file_data.mime_type,
                )
            ))
        if part.inline_data:
            return Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=part.inline_data.data, # type: ignore
                        mimeType=part.inline_data.mime_type,
                    )
                )
            )
        return None

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        task_id = context.task_id or uuid.uuid4().hex
        user_id = None
        if not context.context_id:
            context_id = uuid.uuid4().hex
            session_id = context_id
        else:
            context_id = context.context_id
            if context_id.startswith("users/") and "/sessions/" in context_id:
                context_list = context_id.strip("/").split("/")
                user_id = context_list[1]
                session_id = context_list[-1]
            else:
                session_id = context_id
        if not user_id:
            if (
                context.call_context and
                context.call_context.user and
                context.call_context.user.user_name
            ):
                user_id = context.call_context.user.user_name
            else:
                # When cannot identify a user,
                # use the same value as context id.
                # This way will will connect to the same session in ADK
                user_id = context_id
        logger.info((f"Executing task `{task_id}` in context `{context_id}` "
                    f"for user `{user_id}`"))

        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id
        )
        try:
            session = await self.adk_app.async_get_session(
                user_id=user_id,
                session_id=session_id
            )
        except RuntimeError:
            session = await self.adk_app.async_create_session(
                user_id=user_id,
                session_id=session_id
            )
        if (
            not context.current_task
        ):  # Should be created by DefaultRequestHandler if not existing
            await task_updater.submit()
        if context.message:
            message = types.Content(
                role="user",
                parts=list(filter(lambda p: p is not None, [
                    self._convert_a2a_part_to_genai(p)
                    for p in context.message.parts
                ])) # type: ignore
            )
        else:
            message = types.Content(
                role="user",
            )
        final = False
        stream = self.adk_app.async_stream_query(
            user_id=user_id,
            session_id=session_id,
            message=message.model_dump()
        )
        await task_updater.start_work()
        async for event in stream:
            event_obj = Event.model_validate(event)
            final = event_obj.is_final_response()
            if event_obj.error_code or event_obj.error_message:
                message_chunks = []
                if event_obj.error_code:
                    message_chunks.append(event_obj.error_code)
                if event_obj.error_message:
                    message_chunks.append(event_obj.error_message)
                await task_updater.failed(
                    message=Message(
                        role=Role.agent,
                        contextId=context_id,
                        taskId=task_id,
                        messageId=uuid.uuid4().hex,
                        parts=[
                            Part(
                                root=TextPart(
                                    text=": ".join(message_chunks)
                                )
                            )
                        ]
                    )
                )
                final = True
                break
            function_calls = event_obj.get_function_calls()
            function_responses = event_obj.get_function_responses()
            auth_configs = (event_obj.actions.requested_auth_configs
                            if event_obj.actions else None)
            if auth_configs:
                from a2a.grpc.a2a_pb2 import AuthenticationInfo
                for auth in auth_configs:
                    pass
                await task_updater.update_status(
                    state=TaskState.auth_required
                )

            if function_calls or function_responses or auth_configs:
                if auth_configs or self.transfer_function_calls:
                    await task_updater.add_artifact(
                        parts=[
                            Part(
                                root=DataPart(
                                    data=event_obj.model_dump(
                                        exclude_unset=True
                                    ),
                                    metadata={
                                        "adk_type": "google.adk.events.Event"
                                    }
                                )
                            )
                        ],
                        artifact_id=f"adk_event_{event_obj.id}",
                        name=f"adk_event_{event_obj.id}.json"
                    )
                if auth_configs:
                    await task_updater.update_status(
                        state=TaskState.auth_required
                    )
            if event_obj.actions and event_obj.actions.artifact_delta:
                for key, version in event_obj.actions.artifact_delta.items():
                    artifact = await self.artifact_service.load_artifact( # type: ignore
                        app_name=session.app_name,
                        user_id=session.user_id,
                        session_id=session.id,
                        filename=key,
                        version=version
                    )
                    if artifact:
                        part = self._convert_genai_part_to_a2a(artifact)
                        if part:
                            await task_updater.add_artifact(
                                parts=[
                                    part
                                ],
                                name=key
                            )
            if event_obj.content and event_obj.content.parts:
                message = task_updater.new_agent_message(
                    parts=list(filter(lambda p: p is not None, [
                        self._convert_genai_part_to_a2a(p)
                        for p in event_obj.content.parts
                    ])) # type: ignore
                )
                current_status = TaskState.working
                if context.current_task:
                    current_status = context.current_task.status.state
                await task_updater.update_status(
                    TaskState.completed if final
                    else current_status,
                    message,
                    final
                )

        if context.current_task:
            if not context.current_task.status.state in [
                TaskState.completed,
                TaskState.failed,
                TaskState.canceled,
                TaskState.rejected
            ]:
                await task_updater.complete()

        # Delete task to save memory
        if isinstance(self.task_store, InMemoryTaskStore):
            await self.task_store.delete(task_id)


    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning("Cancelling is not implemented in ADK")
        pass


class A2AStarletteApplicationWithHost(A2AStarletteApplication):
    """This class makes sure the agent's url in the AgentCard has the same
    host, port and schema as in the request.
    """
    @override
    async def _handle_get_agent_card(self, request: Request) -> JSONResponse:
        """Handles GET requests for the agent card endpoint.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A JSONResponse containing the agent card data.
        """
        source_parsed = URL(self.agent_card.url)
        card = self.agent_card.model_copy()
        card.url = str(
            source_parsed.replace(
                hostname=request.url.hostname,
                port=request.url.port,
                scheme=request.url.scheme
            )
        )

        return JSONResponse(
            card.model_dump(mode='json', exclude_none=True)
        )

def adk_as_a2a(
        adk_agent: BaseAgent,
        enable_tracing: Optional[bool] = False,
        session_service_builder: Optional[Callable[..., "BaseSessionService"]] = None,
        artifact_service_builder: Optional[Callable[..., "BaseArtifactService"]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        a2a_agent_version: Optional[str] = None,
        agent_url: str = "http://127.0.0.1:8001/",
        transfer_function_calls: bool = False,
        agent_directory: Optional[str] = None
    ) -> A2AStarletteApplication:
    """Runs an ADK agent as an A2A server."""
    logger.info("Configuring A2A server...")

    if agent_directory:
        agent_json_path = Path(agent_directory) / "agent.json" # type: ignore
    else:
        agent_json_path = None
    if agent_json_path and agent_json_path.exists():
        agent_card_json = agent_json_path.read_text(encoding="utf-8")
        agent_card = AgentCard.model_validate_json(agent_card_json)
    else:
        if hasattr(adk_agent, "input_schema"):
            default_input_mode = "application/json"
        else:
            default_input_mode = "text"
        if hasattr(adk_agent, "output_schema"):
            default_output_mode = "application/json"
        else:
            default_output_mode = "text"

        try:
            agent_card = AgentCard(
                name=adk_agent.name,
                description=adk_agent.description,
                url=agent_url,
                version=a2a_agent_version or "0.0.1",
                capabilities=AgentCapabilities(
                    streaming=True,
                    pushNotifications=False, # Not implemented
                ),
                skills=[
                    AgentSkill(
                        id=adk_agent.name,
                        name=adk_agent.name,
                        description=adk_agent.description,
                        tags=["adk_as_a2a"],
                    )
                ],
                defaultInputModes=[
                    default_input_mode
                ],
                defaultOutputModes=[default_output_mode],
            )
        except Exception:
            logger.exception("Error creating AgentCard")
            raise

    task_store = InMemoryTaskStore()

    try:
        agent_executor = ADKExecutor(
            adk_agent=adk_agent,
            task_store=task_store,
            session_service_builder=session_service_builder,
            artifact_service_builder=artifact_service_builder,
            enable_tracing=enable_tracing,
            env_vars=env_vars,
            transfer_function_calls=transfer_function_calls,
        )
    except Exception:
        logger.exception("Error initializing agent executor")
        raise

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store
    )
    try:
        return A2AStarletteApplicationWithHost(
            agent_card=agent_card,
            http_handler=request_handler,
        )
    except Exception:
        logger.exception("Error initializing A2AStarletteApplication")
        raise

