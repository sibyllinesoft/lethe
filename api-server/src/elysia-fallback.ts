type Handler = (context: {
  request: Request;
  params: Record<string, string>;
  query: Record<string, string>;
  body: unknown;
  set: { status?: number };
}) => any | Promise<any>;

interface Route {
  method: string;
  path: string;
  segments: string[];
  handler: Handler;
}

export class StubElysia {
  private routes: Route[] = [];
  private prefix: string;

  constructor(options: { prefix?: string } = {}) {
    this.prefix = options.prefix ?? '';
  }

  private normalise(path: string): string {
    const fullPath = `${this.prefix}${path}`;
    return fullPath.replace(/\/+$/, '') || '/';
  }

  private addRoute(method: string, path: string, handler: Handler) {
    const normalised = this.normalise(path);
    const segments = normalised.split('/').filter(Boolean);
    this.routes.push({ method, path: normalised, segments, handler });
    return this;
  }

  get(path: string, handler: Handler) {
    return this.addRoute('GET', path, handler);
  }

  post(path: string, handler: Handler) {
    return this.addRoute('POST', path, handler);
  }

  async handle(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const pathname = url.pathname.replace(/\/+$/, '') || '/';
    const segments = pathname.split('/').filter(Boolean);

    for (const route of this.routes) {
      if (route.method !== request.method) continue;
      if (route.segments.length !== segments.length) continue;
      const params: Record<string, string> = {};
      let matched = true;

      for (let i = 0; i < route.segments.length; i++) {
        const candidate = route.segments[i];
        const segment = segments[i];
        if (candidate.startsWith(':')) {
          params[candidate.slice(1)] = decodeURIComponent(segment);
        } else if (candidate !== segment) {
          matched = false;
          break;
        }
      }

      if (!matched) continue;

      const set: { status?: number } = {};
      let body: unknown = undefined;
      if (request.method === 'POST') {
        const contentType = request.headers.get('content-type') ?? '';
        if (contentType.includes('application/json')) {
          body = await request.json();
        } else {
          body = await request.text();
        }
      }

      const result = await route.handler({
        request,
        params,
        query: Object.fromEntries(url.searchParams.entries()),
        body,
        set,
      });

      const status = set.status ?? 200;
      if (result instanceof Response) {
        return result;
      }
      return new Response(JSON.stringify(result), {
        status,
        headers: { 'content-type': 'application/json' },
      });
    }

    return new Response(JSON.stringify({ error: 'Not Found' }), {
      status: 404,
      headers: { 'content-type': 'application/json' },
    });
  }

  async listen({ port = 0, hostname = '127.0.0.1' }: { port?: number; hostname?: string } = {}) {
    const server = Bun.serve({
      port,
      hostname,
      fetch: (request) => this.handle(request),
    });

    return {
      port: server.port,
      hostname: server.hostname,
      stop: () => server.stop(true),
    };
  }
}

export const Elysia = StubElysia;
