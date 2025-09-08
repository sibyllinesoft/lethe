export interface Chunk {
    id: string;
    message_id: string;
    offset_start: number;
    offset_end: number;
    kind: 'prose' | 'code';
    text: string;
    tokens: number;
}
