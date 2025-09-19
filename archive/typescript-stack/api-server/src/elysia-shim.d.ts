declare module 'elysia' {
  export class Elysia {
    constructor(options?: Record<string, unknown>);
    get(path: string, handler: any): this;
    post(path: string, handler: any): this;
    listen(options: { port?: number; hostname?: string }): Promise<{
      port: number;
      hostname: string;
      stop(): void;
    }>;
    handle(request: Request): Promise<Response>;
  }

  export default Elysia;
}
