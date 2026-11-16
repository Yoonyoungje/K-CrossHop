"""
utils/report.py
===============
분석 스크립트(07~16)의 콘솔 출력을 파일로도 영속화하는 tee 헬퍼.

이 스크립트들은 stdout 전용("보고용, downstream 없음")이라 실행 결과가 화면에만
남는다. `--out PATH` 지정 시 start_tee로 stdout을 화면+파일 양쪽에 복제해 결과를 보존한다.
일회성 분석 스크립트용이므로 프로세스 종료까지 유지하고 atexit으로 flush/close 한다.
"""
import sys
import atexit
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def start_tee(path: str) -> str:
    """stdout을 화면+파일로 복제. path가 falsy면 아무것도 안 하고 반환만 한다."""
    if not path:
        return path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, f)
    # flush만 등록(close는 인터프리터 종료가 처리). close하면 종료 시 stdout 최종 flush가
    # 닫힌 파일을 건드려 "I/O operation on closed file" 예외 → atexit 오류 종료가 된다.
    atexit.register(f.flush)
    return path
