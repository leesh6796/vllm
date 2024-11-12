import subprocess
import signal
import sys


def run_command(cmd: str, verbose: bool = False):
    # subprocess.Popen으로 시스템 명령 실행
    process = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )

    try:
        # 프로세스가 종료될 때까지 대기
        process.communicate()
    except KeyboardInterrupt:
        # Ctrl+C 입력 시 KeyboardInterrupt 예외 처리
        if verbose:
            print("\nCtrl+C detected! Terminating the command...")
        process.terminate()  # 프로세스에 SIGTERM 신호 보내기
        process.wait()  # 프로세스가 종료될 때까지 대기
        if verbose:
            print("Command terminated.")
        sys.exit(1)


if __name__ == "__main__":
    # 테스트할 명령어를 여기에 입력 (예: 'sleep 10' 등)
    run_command("sleep 10", verbose=True)
