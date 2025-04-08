import argparse
from pathlib import Path

from src.retrieval.api import create_flask_app
from src.utils.local_logger import LocalLogger


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Run the embedding store API server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval/zef.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Path to the log directory",
        default="logs",
    )
    parser.add_argument(
        "--console_log_level",
        type=str,
        help="Console log level",
        default="INFO",
    )
    parser.add_argument(
        "--file_log_level",
        type=str,
        help="File log level",
        default="DEBUG",
    )

    args = parser.parse_args()
    logger = LocalLogger(
        args.log_dir, "retrieval_server", args.console_log_level, args.file_log_level
    )
    app = create_flask_app(args.config, logger)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
