# MCP Environment

Base environment for connecting a Strands agent to [MCP](https://modelcontextprotocol.io/) servers. Tools exposed by the server are automatically discovered and made available to the agent.

## Setup

No additional dependencies required beyond `strands-env` (`mcp` is a core dependency of `strands-agents`).

## Architecture

`MCPEnvironment` provides shared tool storage (`self._tools`) and default `get_tools()` / `cleanup()` implementations. Subclasses override `reset()` to set up their specific connection method and populate `self._tools` with `MCPTool` subclass instances.

`MCPTool` is an `AgentTool` subclass that handles tool spec building and MCP result parsing. Subclasses implement `_call_tool()` to provide the actual call mechanism (e.g. `ClientSession`, `httpx`, `fastmcp`). Exceptions from `_call_tool()` propagate to the framework's `ToolExecutor` for consistent error handling.

## Subclassing

```python
class MyMCPEnvironment(MCPEnvironment):
    async def reset(self) -> None:
        # Start server, open session, discover tools
        session = ...
        result = await session.list_tools()
        self._tools = [MyMCPTool(tool, session) for tool in result.tools]

    async def cleanup(self) -> None:
        # Close session, kill server, etc.
        ...
        await super().cleanup()  # clears self._tools
```

```python
class MyMCPTool(MCPTool):
    async def _call_tool(self, name, args):
        return await self._session.call_tool(name, args)
```

## Reward

No built-in reward function. Supply a custom `reward_fn`.