import{C as b}from"./CLITerminal-e4bbd15b.js";import{m as t}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./terminal-0325dc00.js";import"./chevron-right-a7178483.js";const z={title:"Components/CLITerminal",component:b,parameters:{layout:"padded",docs:{description:{component:"Terminal-style component that displays CLI command outputs with syntax highlighting and interactive features."}}},tags:["autodocs"],argTypes:{interactive:{control:"boolean"}}},e={args:{outputs:[t.cliOutputs.status,t.cliOutputs.list,t.cliOutputs.analyze],interactive:!1}},a={args:{outputs:[t.cliOutputs.status],interactive:!0}},s={args:{outputs:[t.cliOutputs.status],interactive:!1}},o={args:{outputs:[{command:"prompt-monitor analyze nonexistent_prompt",timestamp:"2024-01-15T14:30:00Z",output:["🔍 Analyzing Prompt: nonexistent_prompt","============================================================","❌ Error: Prompt not found in database","","Available prompts:","  - code_generation_abc123","  - text_analysis_def456","  - creative_writing_ghi789","",'Use "prompt-monitor list-prompts" to see all available prompts.'],status:"error",duration_ms:123}],interactive:!1}},r={args:{outputs:[{command:"prompt-monitor list-prompts --verbose",timestamp:"2024-01-15T14:30:00Z",output:["📋 Tracked Prompts (Detailed)","================================================================================","Prompt ID: code_generation_abc123","  Executions: 45","  Success Rate: 98.5%","  Average Time: 324ms","  Quality Score: 0.892","  Last Used: 2024-01-15T14:30:00","  Template: code_generation","  Variables: {language: python, complexity: medium}","","Prompt ID: text_analysis_def456","  Executions: 32","  Success Rate: 95.2%","  Average Time: 567ms","  Quality Score: 0.845","  Last Used: 2024-01-15T12:15:00","  Template: text_analysis","  Variables: {domain: scientific, length: long}","","Prompt ID: creative_writing_ghi789","  Executions: 28","  Success Rate: 97.1%","  Average Time: 1,245ms","  Quality Score: 0.923","  Last Used: 2024-01-14T16:45:00","  Template: creative_writing","  Variables: {style: narrative, genre: science_fiction}","","📊 Summary Statistics:","  Total Prompts: 3","  Total Executions: 105","  Overall Success Rate: 96.7%","  Average Quality: 0.887"],status:"success",duration_ms:234}],interactive:!1}},n={args:{outputs:[t.cliOutputs.status,t.cliOutputs.list,{command:"prompt-monitor export --format json",timestamp:"2024-01-15T14:35:00Z",output:["📤 Exporting data in JSON format...","✅ Data exported to: exports/prompt_data_20240115_143500.json","📁 File size: 2.3 MB","📊 Records exported: 15,432 executions","🕐 Export completed in 1.2s"],status:"success",duration_ms:1200}],interactive:!0}};var i,m,c;e.parameters={...e.parameters,docs:{...(i=e.parameters)==null?void 0:i.docs,source:{originalSource:`{
  args: {
    outputs: [mockDataset.cliOutputs.status, mockDataset.cliOutputs.list, mockDataset.cliOutputs.analyze],
    interactive: false
  }
}`,...(c=(m=e.parameters)==null?void 0:m.docs)==null?void 0:c.source}}};var u,p,l;a.parameters={...a.parameters,docs:{...(u=a.parameters)==null?void 0:u.docs,source:{originalSource:`{
  args: {
    outputs: [mockDataset.cliOutputs.status],
    interactive: true
  }
}`,...(l=(p=a.parameters)==null?void 0:p.docs)==null?void 0:l.source}}};var d,g,_;s.parameters={...s.parameters,docs:{...(d=s.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    outputs: [mockDataset.cliOutputs.status],
    interactive: false
  }
}`,...(_=(g=s.parameters)==null?void 0:g.docs)==null?void 0:_.source}}};var D,v,x;o.parameters={...o.parameters,docs:{...(D=o.parameters)==null?void 0:D.docs,source:{originalSource:`{
  args: {
    outputs: [{
      command: 'prompt-monitor analyze nonexistent_prompt',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['🔍 Analyzing Prompt: nonexistent_prompt', '============================================================', '❌ Error: Prompt not found in database', '', 'Available prompts:', '  - code_generation_abc123', '  - text_analysis_def456', '  - creative_writing_ghi789', '', 'Use "prompt-monitor list-prompts" to see all available prompts.'],
      status: 'error' as const,
      duration_ms: 123
    }],
    interactive: false
  }
}`,...(x=(v=o.parameters)==null?void 0:v.docs)==null?void 0:x.source}}};var T,y,f;r.parameters={...r.parameters,docs:{...(T=r.parameters)==null?void 0:T.docs,source:{originalSource:`{
  args: {
    outputs: [{
      command: 'prompt-monitor list-prompts --verbose',
      timestamp: '2024-01-15T14:30:00Z',
      output: ['📋 Tracked Prompts (Detailed)', '================================================================================', 'Prompt ID: code_generation_abc123', '  Executions: 45', '  Success Rate: 98.5%', '  Average Time: 324ms', '  Quality Score: 0.892', '  Last Used: 2024-01-15T14:30:00', '  Template: code_generation', '  Variables: {language: python, complexity: medium}', '', 'Prompt ID: text_analysis_def456', '  Executions: 32', '  Success Rate: 95.2%', '  Average Time: 567ms', '  Quality Score: 0.845', '  Last Used: 2024-01-15T12:15:00', '  Template: text_analysis', '  Variables: {domain: scientific, length: long}', '', 'Prompt ID: creative_writing_ghi789', '  Executions: 28', '  Success Rate: 97.1%', '  Average Time: 1,245ms', '  Quality Score: 0.923', '  Last Used: 2024-01-14T16:45:00', '  Template: creative_writing', '  Variables: {style: narrative, genre: science_fiction}', '', '📊 Summary Statistics:', '  Total Prompts: 3', '  Total Executions: 105', '  Overall Success Rate: 96.7%', '  Average Quality: 0.887'],
      status: 'success' as const,
      duration_ms: 234
    }],
    interactive: false
  }
}`,...(f=(y=r.parameters)==null?void 0:y.docs)==null?void 0:f.source}}};var S,C,O;n.parameters={...n.parameters,docs:{...(S=n.parameters)==null?void 0:S.docs,source:{originalSource:`{
  args: {
    outputs: [mockDataset.cliOutputs.status, mockDataset.cliOutputs.list, {
      command: 'prompt-monitor export --format json',
      timestamp: '2024-01-15T14:35:00Z',
      output: ['📤 Exporting data in JSON format...', '✅ Data exported to: exports/prompt_data_20240115_143500.json', '📁 File size: 2.3 MB', '📊 Records exported: 15,432 executions', '🕐 Export completed in 1.2s'],
      status: 'success' as const,
      duration_ms: 1200
    }],
    interactive: true
  }
}`,...(O=(C=n.parameters)==null?void 0:C.docs)==null?void 0:O.source}}};const Q=["Default","Interactive","SingleCommand","ErrorCommand","LongOutput","MultipleCommands"];export{e as Default,o as ErrorCommand,a as Interactive,r as LongOutput,n as MultipleCommands,s as SingleCommand,Q as __namedExportsOrder,z as default};
