"""FastAPI framework, high performance, easy to learn, fast to code, ready for production"""

__version__ = "0.161.1"

from starlette import status as status

from .applications import FastAPI as FastAPI
from .background import BackgroundTasks as BackgroundTasks
from .datastructures import UploadFile as UploadFile
from .dependencies.policy import DependencyPolicyFrame as DependencyPolicyFrame
from .dependencies.policy import dependency_policy_frame as dependency_policy_frame
from .dependencies.policy import (
    get_dependency_policy_stack as get_dependency_policy_stack,
)
from .dependencies.policy import (
    get_effective_cache_partition as get_effective_cache_partition,
)
from .dependencies.policy import (
    get_effective_dependency_overrides as get_effective_dependency_overrides,
)
from .dependencies.policy import (
    get_effective_extra_oauth_scopes as get_effective_extra_oauth_scopes,
)
from .dependencies.policy import pop_dependency_policy as pop_dependency_policy
from .dependencies.policy import push_dependency_policy as push_dependency_policy
from .dependencies.policy import (
    resolve_dependency_override as resolve_dependency_override,
)
from .exceptions import HTTPException as HTTPException
from .exceptions import WebSocketException as WebSocketException
from .param_functions import Body as Body
from .param_functions import Cookie as Cookie
from .param_functions import Depends as Depends
from .param_functions import File as File
from .param_functions import Form as Form
from .param_functions import Header as Header
from .param_functions import Path as Path
from .param_functions import Query as Query
from .param_functions import Security as Security
from .requests import Request as Request
from .responses import Response as Response
from .routing import APIRouter as APIRouter
from .websockets import WebSocket as WebSocket
from .websockets import WebSocketDisconnect as WebSocketDisconnect
