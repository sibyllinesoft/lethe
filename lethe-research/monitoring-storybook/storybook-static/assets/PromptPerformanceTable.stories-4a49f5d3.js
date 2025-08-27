import{P as S}from"./PromptPerformanceTable-40dfe1ae.js";import{m as D,a as Z}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./clock-76184412.js";import"./alert-circle-7b73ffd7.js";const C={title:"Components/PromptPerformanceTable",component:S,parameters:{layout:"padded",docs:{description:{component:"Interactive data table showing prompt performance metrics with sorting, filtering, and search functionality."}}},tags:["autodocs"]},e={args:{data:D.promptPerformance}},t={args:{data:Z(50)}},a={args:{data:Z(3)}},r={args:{data:[{prompt_id:"optimized_code_generation_v3",execution_count:1234,avg_execution_time:89,avg_response_length:1456,avg_quality_score:.96,last_used:"2024-01-15T14:30:00Z",error_count:0,success_rate:100},{prompt_id:"fast_text_summarization_v2",execution_count:987,avg_execution_time:123,avg_response_length:678,avg_quality_score:.94,last_used:"2024-01-15T13:45:00Z",error_count:2,success_rate:99.8},{prompt_id:"efficient_translation_engine",execution_count:756,avg_execution_time:156,avg_response_length:890,avg_quality_score:.92,last_used:"2024-01-15T12:20:00Z",error_count:1,success_rate:99.9}]}},o={args:{data:[{prompt_id:"slow_legacy_analyzer_v1",execution_count:234,avg_execution_time:4567,avg_response_length:1200,avg_quality_score:.67,last_used:"2024-01-14T16:30:00Z",error_count:45,success_rate:80.8},{prompt_id:"unreliable_content_generator",execution_count:123,avg_execution_time:3456,avg_response_length:890,avg_quality_score:.58,last_used:"2024-01-15T09:15:00Z",error_count:32,success_rate:74},{prompt_id:"deprecated_old_template",execution_count:67,avg_execution_time:5678,avg_response_length:1500,avg_quality_score:.45,last_used:"2024-01-13T11:00:00Z",error_count:18,success_rate:73.1}]}},n={args:{data:[]}};var s,c,_;e.parameters={...e.parameters,docs:{...(s=e.parameters)==null?void 0:s.docs,source:{originalSource:`{
  args: {
    data: mockDataset.promptPerformance
  }
}`,...(_=(c=e.parameters)==null?void 0:c.docs)==null?void 0:_.source}}};var i,u,m;t.parameters={...t.parameters,docs:{...(i=t.parameters)==null?void 0:i.docs,source:{originalSource:`{
  args: {
    data: generateMockPromptPerformance(50)
  }
}`,...(m=(u=t.parameters)==null?void 0:u.docs)==null?void 0:m.source}}};var p,g,d;a.parameters={...a.parameters,docs:{...(p=a.parameters)==null?void 0:p.docs,source:{originalSource:`{
  args: {
    data: generateMockPromptPerformance(3)
  }
}`,...(d=(g=a.parameters)==null?void 0:g.docs)==null?void 0:d.source}}};var l,v,x;r.parameters={...r.parameters,docs:{...(l=r.parameters)==null?void 0:l.docs,source:{originalSource:`{
  args: {
    data: [{
      prompt_id: 'optimized_code_generation_v3',
      execution_count: 1234,
      avg_execution_time: 89,
      avg_response_length: 1456,
      avg_quality_score: 0.96,
      last_used: '2024-01-15T14:30:00Z',
      error_count: 0,
      success_rate: 100
    }, {
      prompt_id: 'fast_text_summarization_v2',
      execution_count: 987,
      avg_execution_time: 123,
      avg_response_length: 678,
      avg_quality_score: 0.94,
      last_used: '2024-01-15T13:45:00Z',
      error_count: 2,
      success_rate: 99.8
    }, {
      prompt_id: 'efficient_translation_engine',
      execution_count: 756,
      avg_execution_time: 156,
      avg_response_length: 890,
      avg_quality_score: 0.92,
      last_used: '2024-01-15T12:20:00Z',
      error_count: 1,
      success_rate: 99.9
    }]
  }
}`,...(x=(v=r.parameters)==null?void 0:v.docs)==null?void 0:x.source}}};var f,P,y;o.parameters={...o.parameters,docs:{...(f=o.parameters)==null?void 0:f.docs,source:{originalSource:`{
  args: {
    data: [{
      prompt_id: 'slow_legacy_analyzer_v1',
      execution_count: 234,
      avg_execution_time: 4567,
      avg_response_length: 1200,
      avg_quality_score: 0.67,
      last_used: '2024-01-14T16:30:00Z',
      error_count: 45,
      success_rate: 80.8
    }, {
      prompt_id: 'unreliable_content_generator',
      execution_count: 123,
      avg_execution_time: 3456,
      avg_response_length: 890,
      avg_quality_score: 0.58,
      last_used: '2024-01-15T09:15:00Z',
      error_count: 32,
      success_rate: 74.0
    }, {
      prompt_id: 'deprecated_old_template',
      execution_count: 67,
      avg_execution_time: 5678,
      avg_response_length: 1500,
      avg_quality_score: 0.45,
      last_used: '2024-01-13T11:00:00Z',
      error_count: 18,
      success_rate: 73.1
    }]
  }
}`,...(y=(P=o.parameters)==null?void 0:P.docs)==null?void 0:y.source}}};var h,T,q;n.parameters={...n.parameters,docs:{...(h=n.parameters)==null?void 0:h.docs,source:{originalSource:`{
  args: {
    data: []
  }
}`,...(q=(T=n.parameters)==null?void 0:T.docs)==null?void 0:q.source}}};const I=["Default","LargeDataset","SmallDataset","HighPerformancePrompts","ProblematicPrompts","EmptyState"];export{e as Default,n as EmptyState,r as HighPerformancePrompts,t as LargeDataset,o as ProblematicPrompts,a as SmallDataset,I as __namedExportsOrder,C as default};
