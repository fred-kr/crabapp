import sys


def start() -> None:
    import multiprocessing
    import os
    import time

    multiprocessing.freeze_support()

    from multiprocessing import Condition, Process

    import setproctitle
    import webview

    from crabapp.server import start_dash
    from crabapp.utils import terminate_when_process_dies

    port = os.getenv("PORT", "8050")
    host = os.getenv("HOST", "127.0.0.1")

    server_is_started = Condition()

    # Set the process title.
    setproctitle.setproctitle("crabapp-webview")

    # Spawn the dash process.
    p = Process(target=start_dash, args=(host, port, server_is_started))
    p.start()
    # If the dash process dies, follow along.
    terminate_when_process_dies(p)

    # Wait until dash process is ready.
    with server_is_started:
        server_is_started.wait()
    # FIXME this should not be needed, if server_is_started was triggered after app runs.
    #  idk if that is possible.
    time.sleep(0.2)

    # Allow downloading results as csv file
    webview.settings["ALLOW_DOWNLOADS"] = True
    # Create the webview.
    webview.create_window("CrabApp", f"http://{host}:{port}", width=1600, height=900)
    webview.start()

    # Reached when window is closed.
    p.terminate()
    sys.exit(0)


if __name__ == "__main__":
    start()
