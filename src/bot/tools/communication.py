from src.bot.tools.types import Property, Tool, input_schema_dict

MESSAGE_TOOL = Tool(
    name="message",
    description="Send a message in the current conversation",
    input_schema=input_schema_dict(
        [
            Property(
                name="message_content",
                type="string",
                description="The message content to send",
            ),
        ],
        ["message_content"],
    ),
)

REACT_TOOL = Tool(
    name="react",
    description="React to a message in the current conversation",
    input_schema=input_schema_dict(
        [
            Property(
                name="reaction",
                type="string",
                description="The reaction to send (must be a single emoji)",
            ),
            Property(
                name="username",
                type="string",
                description="The username of the user who sent the message",
            ),
            Property(
                name="identifying_substring",
                type="string",
                description="A subset of the message content that uniquely identifies the message. "
                "Example: 'Hello, world!' -> 'world'",
            ),
        ],
        ["reaction", "username", "identifying_substring"],
    ),
)

REMOVE_REACT_TOOL = Tool(
    name="remove_react",
    description="Remove a reaction from a message in the current conversation",
    input_schema=input_schema_dict(
        [
            Property(
                name="reaction",
                type="string",
                description="The reaction to remove (must be a single emoji)",
            ),
            Property(
                name="username",
                type="string",
                description="The username of the user who sent the message",
            ),
            Property(
                name="identifying_substring",
                type="string",
                description="A subset of the message content that uniquely identifies the message. "
                "Example: 'Hello, world!' -> 'world'",
            ),
        ],
        ["reaction", "username", "identifying_substring"],
    ),
)

DO_NOTHING_TOOL = Tool(
    name="do_nothing",
    description="Do nothing",
    input_schema=input_schema_dict([], []),
)


def is_communication_tool(tool_name: str) -> bool:
    return tool_name in [
        REACT_TOOL.name,
        REMOVE_REACT_TOOL.name,
        MESSAGE_TOOL.name,
        DO_NOTHING_TOOL.name,
    ]
