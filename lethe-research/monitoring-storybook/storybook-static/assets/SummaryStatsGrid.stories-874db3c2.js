import{S as l}from"./SummaryStatsGrid-8f09d531.js";import{m as S}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./DashboardCard-4bfa3037.js";import"./charts-bbe52c79.js";import"./clock-76184412.js";import"./zap-8b467b4b.js";const G={title:"Components/SummaryStatsGrid",component:l,parameters:{layout:"padded",docs:{description:{component:"Grid of summary statistics cards showing key metrics from the prompt monitoring system."}}},tags:["autodocs"]},e={args:{stats:S.summaryStats}},t={args:{stats:{total_executions:45678,unique_prompts:234,success_rate:99.2,avg_execution_time_ms:156,recent_executions_24h:892}}},s={args:{stats:{total_executions:1234,unique_prompts:45,success_rate:87.3,avg_execution_time_ms:2340,recent_executions_24h:23}}},r={args:{stats:{total_executions:156,unique_prompts:12,success_rate:94.8,avg_execution_time_ms:890,recent_executions_24h:34}}};var a,o,n;e.parameters={...e.parameters,docs:{...(a=e.parameters)==null?void 0:a.docs,source:{originalSource:`{
  args: {
    stats: mockDataset.summaryStats
  }
}`,...(n=(o=e.parameters)==null?void 0:o.docs)==null?void 0:n.source}}};var c,m,i;t.parameters={...t.parameters,docs:{...(c=t.parameters)==null?void 0:c.docs,source:{originalSource:`{
  args: {
    stats: {
      total_executions: 45678,
      unique_prompts: 234,
      success_rate: 99.2,
      avg_execution_time_ms: 156,
      recent_executions_24h: 892
    }
  }
}`,...(i=(m=t.parameters)==null?void 0:m.docs)==null?void 0:i.source}}};var u,_,p;s.parameters={...s.parameters,docs:{...(u=s.parameters)==null?void 0:u.docs,source:{originalSource:`{
  args: {
    stats: {
      total_executions: 1234,
      unique_prompts: 45,
      success_rate: 87.3,
      avg_execution_time_ms: 2340,
      recent_executions_24h: 23
    }
  }
}`,...(p=(_=s.parameters)==null?void 0:_.docs)==null?void 0:p.source}}};var d,g,x;r.parameters={...r.parameters,docs:{...(d=r.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    stats: {
      total_executions: 156,
      unique_prompts: 12,
      success_rate: 94.8,
      avg_execution_time_ms: 890,
      recent_executions_24h: 34
    }
  }
}`,...(x=(g=r.parameters)==null?void 0:g.docs)==null?void 0:x.source}}};const H=["Default","HighPerformance","LowPerformance","StartupMetrics"];export{e as Default,t as HighPerformance,s as LowPerformance,r as StartupMetrics,H as __namedExportsOrder,G as default};
