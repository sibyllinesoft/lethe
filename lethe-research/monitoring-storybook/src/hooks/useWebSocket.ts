import { useState, useEffect, useRef, useCallback } from 'react';

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  data: string;
  timestamp: Date;
}

export interface UseWebSocketReturn {
  connectionStatus: ConnectionStatus;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: string) => void;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
}

/**
 * Custom hook for WebSocket connection with automatic reconnection
 * 
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Connection status tracking
 * - Message queuing when disconnected
 * - Error handling and recovery
 * - Memory leak prevention
 */
export const useWebSocket = (
  url?: string,
  enabled = true,
  options: {
    reconnectAttempts?: number;
    reconnectInterval?: number;
    heartbeatInterval?: number;
    protocols?: string[];
  } = {}
): UseWebSocketReturn => {
  const {
    reconnectAttempts = 5,
    heartbeatInterval = 30000,
    protocols
  } = options;

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectCountRef = useRef(0);
  const messageQueueRef = useRef<string[]>([]);
  const mountedRef = useRef(true);

  // Clean up timeouts
  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  // Send heartbeat to keep connection alive
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      
      heartbeatTimeoutRef.current = setTimeout(sendHeartbeat, heartbeatInterval);
    }
  }, [heartbeatInterval]);

  // Process queued messages
  const processMessageQueue = useCallback(() => {
    while (messageQueueRef.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
      const message = messageQueueRef.current.shift();
      if (message) {
        wsRef.current.send(message);
      }
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (!url || !enabled) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    try {
      setConnectionStatus('connecting');
      
      const ws = new WebSocket(url, protocols);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        reconnectCountRef.current = 0;
        
        // Process any queued messages
        processMessageQueue();
        
        // Start heartbeat
        sendHeartbeat();
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        
        // Skip pong messages
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'pong') return;
        } catch {
          // Not JSON, process normally
        }

        setLastMessage({
          data: event.data,
          timestamp: new Date()
        });
      };

      ws.onclose = (event) => {
        if (!mountedRef.current) return;
        
        console.log('WebSocket closed:', event.code, event.reason);
        clearTimeouts();
        
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          // Unexpected close, attempt to reconnect
          setConnectionStatus('connecting');
          const delay = Math.min(1000 * Math.pow(2, reconnectCountRef.current), 30000);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              reconnectCountRef.current++;
              connect();
            }
          }, delay);
        } else {
          setConnectionStatus('disconnected');
        }
      };

      ws.onerror = (error) => {
        if (!mountedRef.current) return;
        
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
    }
  }, [url, enabled, protocols, reconnectAttempts, processMessageQueue, sendHeartbeat, clearTimeouts]);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    clearTimeouts();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setConnectionStatus('disconnected');
    reconnectCountRef.current = 0;
    messageQueueRef.current = [];
  }, [clearTimeouts]);

  // Manual reconnect
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, 100);
  }, [disconnect, connect]);

  // Send message
  const sendMessage = useCallback((message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
    } else {
      // Queue message for when connection is restored
      messageQueueRef.current.push(message);
      
      // Attempt to connect if not already trying
      if (connectionStatus === 'disconnected') {
        connect();
      }
    }
  }, [connectionStatus, connect]);

  // Initial connection
  useEffect(() => {
    if (enabled && url) {
      connect();
    }
    
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [url, enabled, connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      clearTimeouts();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [clearTimeouts]);

  return {
    connectionStatus,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    reconnect
  };
};