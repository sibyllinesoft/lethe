import{c as h,a as t,j as e}from"./createLucideIcon-a1714018.js";import{r as x}from"./index-8b3efc3f.js";import{S as V}from"./SummaryStatsGrid-8f09d531.js";import{T as A}from"./TimelineChart-6880282a.js";import{P as T}from"./PerformanceBubbleChart-01099959.js";import{M as j}from"./ModelComparisonChart-5cefc860.js";import{P as L}from"./PromptPerformanceTable-40dfe1ae.js";import{E,S as z}from"./ExecutionDetailView-24ca68b8.js";import{C as $}from"./CLITerminal-e4bbd15b.js";import{C as p}from"./CLIOutputCard-f8abb1c7.js";import{A as H}from"./DashboardCard-4bfa3037.js";import{T as B}from"./terminal-0325dc00.js";import{m as a}from"./mockData-38ebf9b0.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./clock-76184412.js";import"./zap-8b467b4b.js";import"./generateCategoricalChart-ec1bcc95.js";import"./throttle-32ea0a72.js";import"./mapValues-fb986170.js";import"./tiny-invariant-dd7d57d2.js";import"./isPlainObject-cba39321.js";import"./_baseUniq-e6e71c30.js";import"./Line-c0585f89.js";import"./Scatter-7017bd04.js";import"./alert-circle-7b73ffd7.js";import"./chevron-right-a7178483.js";const F=h("BarChart3",[["path",{d:"M3 3v18h18",key:"1s2lah"}],["path",{d:"M18 17V9",key:"2bz60n"}],["path",{d:"M13 17V5",key:"1frdt8"}],["path",{d:"M8 17v-3",key:"17ska0"}]]),G=h("FileText",[["path",{d:"M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z",key:"1nnpy2"}],["polyline",{points:"14 2 14 8 20 8",key:"1ew0cm"}],["line",{x1:"16",x2:"8",y1:"13",y2:"13",key:"14keom"}],["line",{x1:"16",x2:"8",y1:"17",y2:"17",key:"17nazh"}],["line",{x1:"10",x2:"8",y1:"9",y2:"9",key:"1a5vjj"}]]),J=h("RefreshCw",[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]]),_=({summaryStats:r,timelineData:i,promptPerformance:d,modelComparison:D,executionComparison:k,cliOutputs:o,className:P})=>{const[s,O]=x.useState("overview"),[u,g]=x.useState(!1),q=[{id:"overview",label:"Overview",icon:e(H,{className:"h-5 w-5"}),description:"Summary statistics and trends"},{id:"performance",label:"Performance",icon:e(F,{className:"h-5 w-5"}),description:"Charts and analytics"},{id:"executions",label:"Executions",icon:e(G,{className:"h-5 w-5"}),description:"Detailed execution data"},{id:"cli",label:"CLI",icon:e(B,{className:"h-5 w-5"}),description:"Command-line interface"},{id:"settings",label:"Settings",icon:e(z,{className:"h-5 w-5"}),description:"Configuration options"}],I=async()=>{g(!0),await new Promise(n=>setTimeout(n,1e3)),g(!1)};return t("div",{className:`min-h-screen bg-gray-50 ${P}`,children:[t("header",{className:"bg-white border-b border-gray-200 sticky top-0 z-40",children:[e("div",{className:"px-6 py-4",children:t("div",{className:"flex items-center justify-between",children:[t("div",{children:[e("h1",{className:"text-2xl font-bold text-gray-900 flex items-center",children:"🔍 Lethe Prompt Monitor"}),e("p",{className:"text-sm text-gray-600 mt-1",children:"Real-time monitoring and analytics for prompt executions"})]}),t("button",{onClick:I,disabled:u,className:"flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors",children:[e(J,{className:`h-4 w-4 mr-2 ${u?"animate-spin":""}`}),u?"Refreshing...":"Refresh"]})]})}),e("nav",{className:"px-6",children:e("div",{className:"flex space-x-8 overflow-x-auto",children:q.map(n=>t("button",{onClick:()=>O(n.id),className:`flex items-center space-x-2 px-1 py-4 border-b-2 text-sm font-medium whitespace-nowrap transition-colors ${s===n.id?"border-blue-500 text-blue-600":"border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"}`,children:[n.icon,e("span",{children:n.label})]},n.id))})})]}),t("main",{className:"px-6 py-6",children:[s==="overview"&&t("div",{className:"space-y-6",children:[t("section",{children:[e("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Summary Statistics"}),e(V,{stats:r})]}),e("section",{children:e(A,{data:i})}),t("section",{className:"grid grid-cols-1 md:grid-cols-3 gap-4",children:[t("div",{className:"bg-white p-6 rounded-lg shadow-sm border border-gray-200",children:[e("h3",{className:"text-lg font-medium text-gray-900 mb-2",children:"Recent Status"}),e(p,{output:o.status,compact:!0})]}),t("div",{className:"bg-white p-6 rounded-lg shadow-sm border border-gray-200",children:[e("h3",{className:"text-lg font-medium text-gray-900 mb-2",children:"Top Prompts"}),e("div",{className:"space-y-2",children:d.slice(0,3).map((n,R)=>t("div",{className:"flex items-center justify-between",children:[e("span",{className:"text-sm text-gray-600 truncate",children:n.prompt_id}),e("span",{className:"text-xs text-gray-500",children:n.execution_count})]},R))})]}),t("div",{className:"bg-white p-6 rounded-lg shadow-sm border border-gray-200",children:[e("h3",{className:"text-lg font-medium text-gray-900 mb-2",children:"System Health"}),t("div",{className:"space-y-2",children:[t("div",{className:"flex items-center justify-between",children:[e("span",{className:"text-sm text-gray-600",children:"Success Rate"}),t("span",{className:"text-sm font-medium text-green-600",children:[r.success_rate.toFixed(1),"%"]})]}),t("div",{className:"flex items-center justify-between",children:[e("span",{className:"text-sm text-gray-600",children:"Avg Response"}),t("span",{className:"text-sm font-medium text-blue-600",children:[Math.round(r.avg_execution_time_ms),"ms"]})]}),t("div",{className:"flex items-center justify-between",children:[e("span",{className:"text-sm text-gray-600",children:"24h Activity"}),e("span",{className:"text-sm font-medium text-purple-600",children:r.recent_executions_24h})]})]})]})]})]}),s==="performance"&&t("div",{className:"space-y-6",children:[t("section",{children:[e("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Performance Analytics"}),t("div",{className:"grid grid-cols-1 xl:grid-cols-2 gap-6",children:[e(T,{data:d}),e(j,{data:D})]})]}),e("section",{children:e(L,{data:d})})]}),s==="executions"&&e("div",{className:"space-y-6",children:t("section",{children:[e("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Execution Details"}),e(E,{comparison:k})]})}),s==="cli"&&t("div",{className:"space-y-6",children:[t("section",{children:[e("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Command Line Interface"}),e($,{outputs:[o.status,o.list,o.analyze],interactive:!0})]}),t("section",{children:[e("h3",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Recent Commands"}),t("div",{className:"grid grid-cols-1 lg:grid-cols-2 gap-4",children:[e(p,{output:o.list}),e(p,{output:o.analyze})]})]})]}),s==="settings"&&e("div",{className:"space-y-6",children:t("section",{children:[e("h2",{className:"text-lg font-semibold text-gray-900 mb-4",children:"Configuration Settings"}),e("div",{className:"bg-white rounded-lg shadow-sm border border-gray-200 p-6",children:t("div",{className:"space-y-6",children:[t("div",{children:[e("h3",{className:"text-base font-medium text-gray-900 mb-4",children:"Database Configuration"}),t("div",{className:"grid grid-cols-1 md:grid-cols-2 gap-4",children:[t("div",{children:[e("label",{className:"block text-sm font-medium text-gray-700 mb-2",children:"Database Path"}),e("input",{type:"text",value:"experiments/prompt_tracking.db",className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500",readOnly:!0})]}),t("div",{children:[e("label",{className:"block text-sm font-medium text-gray-700 mb-2",children:"Auto-cleanup (days)"}),e("input",{type:"number",value:"30",className:"w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"})]})]})]}),t("div",{children:[e("h3",{className:"text-base font-medium text-gray-900 mb-4",children:"Monitoring Settings"}),t("div",{className:"space-y-4",children:[t("div",{className:"flex items-center justify-between",children:[t("div",{children:[e("h4",{className:"text-sm font-medium text-gray-900",children:"Real-time Updates"}),e("p",{className:"text-sm text-gray-500",children:"Automatically refresh data every 30 seconds"})]}),e("input",{type:"checkbox",className:"h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded",defaultChecked:!0})]}),t("div",{className:"flex items-center justify-between",children:[t("div",{children:[e("h4",{className:"text-sm font-medium text-gray-900",children:"Performance Alerts"}),e("p",{className:"text-sm text-gray-500",children:"Notify when execution time exceeds threshold"})]}),e("input",{type:"checkbox",className:"h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded",defaultChecked:!0})]})]})]}),t("div",{children:[e("h3",{className:"text-base font-medium text-gray-900 mb-4",children:"Export Options"}),t("div",{className:"flex space-x-4",children:[e("button",{className:"px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors",children:"Export as JSON"}),e("button",{className:"px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors",children:"Export as CSV"}),e("button",{className:"px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors",children:"Generate Report"})]})]})]})})]})})]})]})};try{_.displayName="MonitoringDashboard",_.__docgenInfo={description:`Main monitoring dashboard component
Combines all monitoring visualizations into a comprehensive interface
Based on the PromptDashboard class functionality`,displayName:"MonitoringDashboard",props:{summaryStats:{defaultValue:null,description:"",name:"summaryStats",required:!0,type:{name:"SummaryStats"}},timelineData:{defaultValue:null,description:"",name:"timelineData",required:!0,type:{name:"TimelineDataPoint[]"}},promptPerformance:{defaultValue:null,description:"",name:"promptPerformance",required:!0,type:{name:"PromptPerformance[]"}},modelComparison:{defaultValue:null,description:"",name:"modelComparison",required:!0,type:{name:"ModelComparison[]"}},executionComparison:{defaultValue:null,description:"",name:"executionComparison",required:!0,type:{name:"ExecutionComparison"}},cliOutputs:{defaultValue:null,description:"",name:"cliOutputs",required:!0,type:{name:"{ status: CLIOutput; list: CLIOutput; analyze: CLIOutput; }"}},className:{defaultValue:null,description:"",name:"className",required:!1,type:{name:"string | undefined"}}}}}catch{}const fe={title:"Pages/MonitoringDashboard",component:_,parameters:{layout:"fullscreen",docs:{description:{component:"Complete monitoring dashboard interface combining all components into a tabbed, responsive dashboard for prompt monitoring analytics."}}},tags:["autodocs"]},c={args:{summaryStats:a.summaryStats,timelineData:a.timelineData,promptPerformance:a.promptPerformance,modelComparison:a.modelComparison,executionComparison:a.executionComparison,cliOutputs:a.cliOutputs}},m={args:{summaryStats:{total_executions:125678,unique_prompts:456,success_rate:98.7,avg_execution_time_ms:189,recent_executions_24h:2341},timelineData:[{date:"2024-01-09",total_executions:1234,avg_execution_time:156,avg_response_length:1100,errors:12},{date:"2024-01-10",total_executions:1456,avg_execution_time:167,avg_response_length:1200,errors:8},{date:"2024-01-11",total_executions:1789,avg_execution_time:145,avg_response_length:1150,errors:5},{date:"2024-01-12",total_executions:2012,avg_execution_time:134,avg_response_length:1080,errors:3},{date:"2024-01-13",total_executions:1987,avg_execution_time:142,avg_response_length:1220,errors:7},{date:"2024-01-14",total_executions:2134,avg_execution_time:128,avg_response_length:1300,errors:2},{date:"2024-01-15",total_executions:2341,avg_execution_time:119,avg_response_length:1250,errors:1}],promptPerformance:Array.from({length:15},(r,i)=>({prompt_id:`high_volume_prompt_${String(i+1).padStart(3,"0")}`,execution_count:Math.floor(Math.random()*1e3)+500,avg_execution_time:Math.floor(Math.random()*300)+100,avg_response_length:Math.floor(Math.random()*800)+600,avg_quality_score:.85+Math.random()*.15,last_used:new Date(Date.now()-Math.random()*7*24*60*60*1e3).toISOString(),error_count:Math.floor(Math.random()*10),success_rate:95+Math.random()*5})),modelComparison:a.modelComparison,executionComparison:a.executionComparison,cliOutputs:a.cliOutputs}},l={args:{summaryStats:{total_executions:342,unique_prompts:15,success_rate:92.1,avg_execution_time_ms:567,recent_executions_24h:23},timelineData:[{date:"2024-01-09",total_executions:12,avg_execution_time:890,avg_response_length:800,errors:2},{date:"2024-01-10",total_executions:18,avg_execution_time:756,avg_response_length:950,errors:1},{date:"2024-01-11",total_executions:25,avg_execution_time:634,avg_response_length:1100,errors:0},{date:"2024-01-12",total_executions:31,avg_execution_time:598,avg_response_length:1050,errors:3},{date:"2024-01-13",total_executions:28,avg_execution_time:612,avg_response_length:980,errors:1},{date:"2024-01-14",total_executions:34,avg_execution_time:534,avg_response_length:1200,errors:0},{date:"2024-01-15",total_executions:23,avg_execution_time:567,avg_response_length:1150,errors:1}],promptPerformance:Array.from({length:5},(r,i)=>({prompt_id:`startup_prompt_${String(i+1).padStart(2,"0")}`,execution_count:Math.floor(Math.random()*50)+10,avg_execution_time:Math.floor(Math.random()*500)+300,avg_response_length:Math.floor(Math.random()*600)+400,avg_quality_score:.75+Math.random()*.25,last_used:new Date(Date.now()-Math.random()*7*24*60*60*1e3).toISOString(),error_count:Math.floor(Math.random()*5),success_rate:85+Math.random()*15})),modelComparison:[{model_name:"gpt-3.5-turbo",execution_count:234,avg_execution_time:456,avg_response_length:890,avg_quality_score:.82,error_count:12},{model_name:"claude-3-haiku-20240307",execution_count:108,avg_execution_time:234,avg_response_length:1100,avg_quality_score:.87,error_count:3}],executionComparison:a.executionComparison,cliOutputs:a.cliOutputs}};var v,y,f;c.parameters={...c.parameters,docs:{...(v=c.parameters)==null?void 0:v.docs,source:{originalSource:`{
  args: {
    summaryStats: mockDataset.summaryStats,
    timelineData: mockDataset.timelineData,
    promptPerformance: mockDataset.promptPerformance,
    modelComparison: mockDataset.modelComparison,
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
}`,...(f=(y=c.parameters)==null?void 0:y.docs)==null?void 0:f.source}}};var b,N,M;m.parameters={...m.parameters,docs:{...(b=m.parameters)==null?void 0:b.docs,source:{originalSource:`{
  args: {
    summaryStats: {
      total_executions: 125678,
      unique_prompts: 456,
      success_rate: 98.7,
      avg_execution_time_ms: 189,
      recent_executions_24h: 2341
    },
    timelineData: [{
      date: '2024-01-09',
      total_executions: 1234,
      avg_execution_time: 156,
      avg_response_length: 1100,
      errors: 12
    }, {
      date: '2024-01-10',
      total_executions: 1456,
      avg_execution_time: 167,
      avg_response_length: 1200,
      errors: 8
    }, {
      date: '2024-01-11',
      total_executions: 1789,
      avg_execution_time: 145,
      avg_response_length: 1150,
      errors: 5
    }, {
      date: '2024-01-12',
      total_executions: 2012,
      avg_execution_time: 134,
      avg_response_length: 1080,
      errors: 3
    }, {
      date: '2024-01-13',
      total_executions: 1987,
      avg_execution_time: 142,
      avg_response_length: 1220,
      errors: 7
    }, {
      date: '2024-01-14',
      total_executions: 2134,
      avg_execution_time: 128,
      avg_response_length: 1300,
      errors: 2
    }, {
      date: '2024-01-15',
      total_executions: 2341,
      avg_execution_time: 119,
      avg_response_length: 1250,
      errors: 1
    }],
    promptPerformance: Array.from({
      length: 15
    }, (_, i) => ({
      prompt_id: \`high_volume_prompt_\${String(i + 1).padStart(3, '0')}\`,
      execution_count: Math.floor(Math.random() * 1000) + 500,
      avg_execution_time: Math.floor(Math.random() * 300) + 100,
      avg_response_length: Math.floor(Math.random() * 800) + 600,
      avg_quality_score: 0.85 + Math.random() * 0.15,
      last_used: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      error_count: Math.floor(Math.random() * 10),
      success_rate: 95 + Math.random() * 5
    })),
    modelComparison: mockDataset.modelComparison,
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
}`,...(M=(N=m.parameters)==null?void 0:N.docs)==null?void 0:M.source}}};var C,w,S;l.parameters={...l.parameters,docs:{...(C=l.parameters)==null?void 0:C.docs,source:{originalSource:`{
  args: {
    summaryStats: {
      total_executions: 342,
      unique_prompts: 15,
      success_rate: 92.1,
      avg_execution_time_ms: 567,
      recent_executions_24h: 23
    },
    timelineData: [{
      date: '2024-01-09',
      total_executions: 12,
      avg_execution_time: 890,
      avg_response_length: 800,
      errors: 2
    }, {
      date: '2024-01-10',
      total_executions: 18,
      avg_execution_time: 756,
      avg_response_length: 950,
      errors: 1
    }, {
      date: '2024-01-11',
      total_executions: 25,
      avg_execution_time: 634,
      avg_response_length: 1100,
      errors: 0
    }, {
      date: '2024-01-12',
      total_executions: 31,
      avg_execution_time: 598,
      avg_response_length: 1050,
      errors: 3
    }, {
      date: '2024-01-13',
      total_executions: 28,
      avg_execution_time: 612,
      avg_response_length: 980,
      errors: 1
    }, {
      date: '2024-01-14',
      total_executions: 34,
      avg_execution_time: 534,
      avg_response_length: 1200,
      errors: 0
    }, {
      date: '2024-01-15',
      total_executions: 23,
      avg_execution_time: 567,
      avg_response_length: 1150,
      errors: 1
    }],
    promptPerformance: Array.from({
      length: 5
    }, (_, i) => ({
      prompt_id: \`startup_prompt_\${String(i + 1).padStart(2, '0')}\`,
      execution_count: Math.floor(Math.random() * 50) + 10,
      avg_execution_time: Math.floor(Math.random() * 500) + 300,
      avg_response_length: Math.floor(Math.random() * 600) + 400,
      avg_quality_score: 0.75 + Math.random() * 0.25,
      last_used: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
      error_count: Math.floor(Math.random() * 5),
      success_rate: 85 + Math.random() * 15
    })),
    modelComparison: [{
      model_name: 'gpt-3.5-turbo',
      execution_count: 234,
      avg_execution_time: 456,
      avg_response_length: 890,
      avg_quality_score: 0.82,
      error_count: 12
    }, {
      model_name: 'claude-3-haiku-20240307',
      execution_count: 108,
      avg_execution_time: 234,
      avg_response_length: 1100,
      avg_quality_score: 0.87,
      error_count: 3
    }],
    executionComparison: mockDataset.executionComparison,
    cliOutputs: mockDataset.cliOutputs
  }
}`,...(S=(w=l.parameters)==null?void 0:w.docs)==null?void 0:S.source}}};const be=["Default","HighVolumeSystem","StartupSystem"];export{c as Default,m as HighVolumeSystem,l as StartupSystem,be as __namedExportsOrder,fe as default};
