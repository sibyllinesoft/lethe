import{C as I}from"./CLIOutputCard-f8abb1c7.js";import{m as p}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./clock-76184412.js";import"./terminal-0325dc00.js";import"./alert-circle-7b73ffd7.js";const q={title:"Components/CLIOutputCard",component:I,parameters:{layout:"padded",docs:{description:{component:"Card component for displaying individual CLI command outputs with syntax highlighting and copy functionality."}}},tags:["autodocs"],argTypes:{compact:{control:"boolean"}}},t={args:{output:p.cliOutputs.status,compact:!1}},o={args:{output:p.cliOutputs.status,compact:!0}},e={args:{output:p.cliOutputs.list,compact:!1}},a={args:{output:p.cliOutputs.analyze,compact:!1}},r={args:{output:{command:"prompt-monitor analyze nonexistent_prompt",timestamp:"2024-01-15T14:30:00Z",output:["🔍 Analyzing Prompt: nonexistent_prompt","============================================================",'❌ Error: Prompt "nonexistent_prompt" not found in database',"","Available prompts:","  - code_generation_abc123","  - text_analysis_def456","  - creative_writing_ghi789","  - question_answering_jkl012","",'Use "prompt-monitor list-prompts" to see all available prompts.','Use "prompt-monitor --help" for more commands.'],status:"error",duration_ms:156},compact:!1}},s={args:{output:{command:"prompt-monitor cleanup --days 7",timestamp:"2024-01-15T14:30:00Z",output:["🧹 Cleaning up data older than 7 days (2024-01-08)","⚠️ Warning: This will permanently delete 2,341 execution records","⚠️ Warning: Some prompt analytics may be affected","","Are you sure you want to continue? (y/N)","","💡 Consider using --days 30 to retain more historical data","💡 Use --dry-run to preview what will be deleted"],status:"warning",duration_ms:89},compact:!1}},n={args:{output:{command:"prompt-monitor export --format json --verbose",timestamp:"2024-01-15T14:30:00Z",output:["📤 Exporting prompt execution data...","================================================================================","📊 Export Configuration:","  Format: JSON","  Include metadata: Yes","  Include environment variables: Yes","  Include error details: Yes","  Date range: All time","","🔍 Scanning database...","  Found 15,432 execution records","  Found 127 unique prompts","  Found 1,234 comparison records","","📝 Processing records...","  [████████████████████████████████████████] 100% (15,432/15,432)","","💾 Writing to file...","  Output file: exports/prompt_data_20240115_143000.json","  File size: 2.34 MB","  Compression: Enabled (67% reduction)","","✅ Export completed successfully!","📁 File location: /home/user/lethe/exports/prompt_data_20240115_143000.json","📊 Summary:","  Total executions: 15,432","  Date range: 2023-12-01 to 2024-01-15","  Export duration: 3.2 seconds","  Average processing rate: 4,822 records/second"],status:"success",duration_ms:3200},compact:!1}},u={args:{output:{command:"prompt-monitor --version",timestamp:"2024-01-15T14:30:00Z",output:["Lethe Prompt Monitor v1.2.3","Built on 2024-01-10","Python 3.11.0"],status:"success",duration_ms:23},compact:!1}};var m,i,c;t.parameters={...t.parameters,docs:{...(m=t.parameters)==null?void 0:m.docs,source:{originalSource:`{
  args: {
    output: mockDataset.cliOutputs.status,
    compact: false
  }
}`,...(c=(i=t.parameters)==null?void 0:i.docs)==null?void 0:c.source}}};var d,l,g;o.parameters={...o.parameters,docs:{...(d=o.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    output: mockDataset.cliOutputs.status,
    compact: true
  }
}`,...(g=(l=o.parameters)==null?void 0:l.docs)==null?void 0:g.source}}};var D,_,y;e.parameters={...e.parameters,docs:{...(D=e.parameters)==null?void 0:D.docs,source:{originalSource:`{
  args: {
    output: mockDataset.cliOutputs.list,
    compact: false
  }
}`,...(y=(_=e.parameters)==null?void 0:_.docs)==null?void 0:y.source}}};var f,C,x;a.parameters={...a.parameters,docs:{...(f=a.parameters)==null?void 0:f.docs,source:{originalSource:`{
  args: {
    output: mockDataset.cliOutputs.analyze,
    compact: false
  }
}`,...(x=(C=a.parameters)==null?void 0:C.docs)==null?void 0:x.source}}};var h,b,v;r.parameters={...r.parameters,docs:{...(h=r.parameters)==null?void 0:h.docs,source:{originalSource:`{
  args: {
    output: {
      command: 'prompt-monitor analyze nonexistent_prompt',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['🔍 Analyzing Prompt: nonexistent_prompt', '============================================================', '❌ Error: Prompt "nonexistent_prompt" not found in database', '', 'Available prompts:', '  - code_generation_abc123', '  - text_analysis_def456', '  - creative_writing_ghi789', '  - question_answering_jkl012', '', 'Use "prompt-monitor list-prompts" to see all available prompts.', 'Use "prompt-monitor --help" for more commands.'],
      status: 'error' as const,
      duration_ms: 156
    },
    compact: false
  }
}`,...(v=(b=r.parameters)==null?void 0:b.docs)==null?void 0:v.source}}};var E,F,A;s.parameters={...s.parameters,docs:{...(E=s.parameters)==null?void 0:E.docs,source:{originalSource:`{
  args: {
    output: {
      command: 'prompt-monitor cleanup --days 7',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['🧹 Cleaning up data older than 7 days (2024-01-08)', '⚠️ Warning: This will permanently delete 2,341 execution records', '⚠️ Warning: Some prompt analytics may be affected', '', 'Are you sure you want to continue? (y/N)', '', '💡 Consider using --days 30 to retain more historical data', '💡 Use --dry-run to preview what will be deleted'],
      status: 'warning' as const,
      duration_ms: 89
    },
    compact: false
  }
}`,...(A=(F=s.parameters)==null?void 0:F.docs)==null?void 0:A.source}}};var w,O,S;n.parameters={...n.parameters,docs:{...(w=n.parameters)==null?void 0:w.docs,source:{originalSource:`{
  args: {
    output: {
      command: 'prompt-monitor export --format json --verbose',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['📤 Exporting prompt execution data...', '================================================================================', '📊 Export Configuration:', '  Format: JSON', '  Include metadata: Yes', '  Include environment variables: Yes', '  Include error details: Yes', '  Date range: All time', '', '🔍 Scanning database...', '  Found 15,432 execution records', '  Found 127 unique prompts', '  Found 1,234 comparison records', '', '📝 Processing records...', '  [████████████████████████████████████████] 100% (15,432/15,432)', '', '💾 Writing to file...', '  Output file: exports/prompt_data_20240115_143000.json', '  File size: 2.34 MB', '  Compression: Enabled (67% reduction)', '', '✅ Export completed successfully!', '📁 File location: /home/user/lethe/exports/prompt_data_20240115_143000.json', '📊 Summary:', '  Total executions: 15,432', '  Date range: 2023-12-01 to 2024-01-15', '  Export duration: 3.2 seconds', '  Average processing rate: 4,822 records/second'],
      status: 'success' as const,
      duration_ms: 3200
    },
    compact: false
  }
}`,...(S=(O=n.parameters)==null?void 0:O.docs)==null?void 0:S.source}}};var T,P,z;u.parameters={...u.parameters,docs:{...(T=u.parameters)==null?void 0:T.docs,source:{originalSource:`{
  args: {
    output: {
      command: 'prompt-monitor --version',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['Lethe Prompt Monitor v1.2.3', 'Built on 2024-01-10', 'Python 3.11.0'],
      status: 'success' as const,
      duration_ms: 23
    },
    compact: false
  }
}`,...(z=(P=u.parameters)==null?void 0:P.docs)==null?void 0:z.source}}};const M=["Default","Compact","ListPrompts","AnalyzeCommand","ErrorCommand","WarningCommand","LongOutput","FastCommand"];export{a as AnalyzeCommand,o as Compact,t as Default,r as ErrorCommand,u as FastCommand,e as ListPrompts,n as LongOutput,s as WarningCommand,M as __namedExportsOrder,q as default};
