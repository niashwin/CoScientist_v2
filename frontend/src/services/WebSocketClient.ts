export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export interface WebSocketOptions {
  maxReconnectAttempts?: number;
  reconnectDelay?: number;
  messageHandlers?: Record<string, (message: WebSocketMessage) => void>;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onReconnect?: () => void;
  onError?: (error: Event) => void;
}

export class ResilientWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private clientId: string;
  private lastMessageId: string | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private messageHandlers = new Map<string, (message: WebSocketMessage) => void>();
  private messageBuffer: WebSocketMessage[] = [];
  private isReconnecting = false;
  private heartbeatInterval: number | null = null;
  
  // Event handlers
  private onConnect: () => void;
  private onDisconnect: () => void;
  private onReconnect: () => void;
  private onError: (error: Event) => void;
  
  constructor(url: string, options: WebSocketOptions = {}) {
    this.url = url;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectDelay = options.reconnectDelay || 1000;
    this.onConnect = options.onConnect || (() => {});
    this.onDisconnect = options.onDisconnect || (() => {});
    this.onReconnect = options.onReconnect || (() => {});
    this.onError = options.onError || (() => {});
    
    // Get or create client ID
    this.clientId = this.getOrCreateClientId();
    this.lastMessageId = localStorage.getItem('ws_last_message_id');
    
    // Register handlers from options BEFORE connecting
    if (options.messageHandlers) {
      for (const [type, handler] of Object.entries(options.messageHandlers)) {
        this.on(type, handler);
      }
    }
    
    this.connect();
  }
  
  private getOrCreateClientId(): string {
    let clientId = localStorage.getItem('ws_client_id');
    if (!clientId) {
      clientId = crypto.randomUUID();
      localStorage.setItem('ws_client_id', clientId);
    }
    return clientId;
  }
  
  private connect(): void {
    try {
      // Create WebSocket connection
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        
        // Send connection info
        this.send({
          type: 'connect',
          client_id: this.clientId,
          last_message_id: this.lastMessageId
        });
        
        // Start heartbeat
        this.startHeartbeat();
        
        // Send any buffered messages
        this.flushMessageBuffer();
        
        if (this.lastMessageId) {
          this.onReconnect();
        } else {
          this.onConnect();
        }
      };
      
      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.stopHeartbeat();
        this.onDisconnect();
        
        // Attempt reconnection if not a clean close
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.onError(error);
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }
  
  private scheduleReconnect(): void {
    if (this.isReconnecting) return;
    
    this.isReconnecting = true;
    this.reconnectAttempts++;
    
    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) + Math.random() * 1000,
      30000 // Max 30 seconds
    );
    
    console.log(`Reconnecting in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect();
    }, delay);
  }
  
  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Every 30 seconds
  }
  
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
  
  private handleMessage(message: WebSocketMessage): void {
    // Add debugging to see all messages
    console.log('WebSocket received message:', message);
    console.log('Available handlers:', Array.from(this.messageHandlers.keys()));
    
    // Track last message ID
    if (message.message_id) {
      this.lastMessageId = message.message_id;
      localStorage.setItem('ws_last_message_id', message.message_id);
    }
    
    // Handle system messages
    switch (message.type) {
      case 'connection_established':
        console.log('Connection established:', message.client_id);
        return;
        
      case 'reconnection_successful':
        console.log('Reconnection successful:', message.reconnection_count);
        return;
        
      case 'pong':
        // Heartbeat response
        return;
        
      case 'buffered_messages_start':
        console.log(`Receiving ${message.count} buffered messages`);
        return;
        
      case 'buffered_messages_complete':
        console.log('All buffered messages received');
        return;
    }
    
    // Handle application messages
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      console.log(`Calling handler for message type: ${message.type}`, message);
      try {
        handler(message);
        console.log(`Handler for ${message.type} completed successfully`);
      } catch (error) {
        console.error(`Error in handler for ${message.type}:`, error);
      }
    } else {
      console.warn('No handler for message type:', message.type, message);
    }
  }
  
  public send(data: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      // Buffer message for sending after reconnection
      this.messageBuffer.push(data);
    }
  }
  
  private flushMessageBuffer(): void {
    while (this.messageBuffer.length > 0) {
      const message = this.messageBuffer.shift();
      if (message) {
        this.send(message);
      }
    }
  }
  
  public on(messageType: string, handler: (message: WebSocketMessage) => void): void {
    console.log(`Registering handler for message type: ${messageType}`);
    this.messageHandlers.set(messageType, handler);
    console.log(`Total handlers registered: ${this.messageHandlers.size}`);
  }
  
  public off(messageType: string): void {
    this.messageHandlers.delete(messageType);
  }
  
  public close(): void {
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent reconnection
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
    }
  }
  
  public get connectionStatus(): 'connecting' | 'connected' | 'disconnected' | 'reconnecting' {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return this.isReconnecting ? 'reconnecting' : 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
      default:
        return 'disconnected';
    }
  }
} 