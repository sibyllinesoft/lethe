declare module '@monaco-editor/react' {
  const DiffEditor: any;
  export { DiffEditor };
  export default DiffEditor;
}

declare module 'monaco-editor' {
  export const editor: any;
}
