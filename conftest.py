"""pytest 가 프로젝트 루트를 import 경로에 포함하도록 한다.

모듈이 루트에 평면 배치되어 있으므로(SPEC.md 7절) 테스트에서
``from contracts import Box`` 형태로 직접 import 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
