import { startServer } from './server';

startServer().then((server) => {
  console.log(`🦊 Dev server running at http://${server.host}:${server.port}`);
});
