from retrieval.api import create_flask_app
import argparse


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

    args = parser.parse_args()

    app = create_flask_app(args.config)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
