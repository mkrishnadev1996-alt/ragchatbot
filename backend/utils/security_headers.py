from collections.abc import Iterable


class SecurityHeadersMiddleware:
    """Adds fixed response headers (pure ASGI, so streamed bodies pass straight through).

    Headers are only set when the response doesn't already carry them. Paths in
    `skip_csp_paths` don't get the Content-Security-Policy header.
    """

    def __init__(self, app, headers: dict[str, str], skip_csp_paths: Iterable[str] = ()):
        self.app = app
        self.headers = {k.lower().encode(): v.encode() for k, v in headers.items()}
        self.skip_csp_paths = frozenset(skip_csp_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        skip_csp = scope.get("path") in self.skip_csp_paths

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                present = {name.lower() for name, _ in raw}
                for name, value in self.headers.items():
                    if name in present or (skip_csp and name == b"content-security-policy"):
                        continue
                    raw.append((name, value))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_with_headers)
