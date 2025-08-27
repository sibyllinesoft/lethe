import{P as w}from"./PerformanceBubbleChart-01099959.js";import{m as T,a as Z}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./generateCategoricalChart-ec1bcc95.js";import"./throttle-32ea0a72.js";import"./mapValues-fb986170.js";import"./tiny-invariant-dd7d57d2.js";import"./isPlainObject-cba39321.js";import"./_baseUniq-e6e71c30.js";import"./Scatter-7017bd04.js";const A={title:"Components/PerformanceBubbleChart",component:w,parameters:{layout:"padded",docs:{description:{component:"Bubble chart visualization showing prompt performance with execution time vs count, bubble size representing success rate, and color indicating quality score."}}},tags:["autodocs"],argTypes:{height:{control:{type:"range",min:400,max:800,step:50}}}},e={args:{data:T.promptPerformance,height:500}},t={args:{data:Z(25),height:600}},r={args:{data:Z(5),height:400}},a={args:{data:T.promptPerformance.slice(0,8),height:400}},o={args:{data:[{prompt_id:"optimized_code_gen_v3",execution_count:234,avg_execution_time:156,avg_response_length:1200,avg_quality_score:.95,last_used:"2024-01-15T14:30:00Z",error_count:0,success_rate:100},{prompt_id:"fast_text_analysis_v2",execution_count:187,avg_execution_time:89,avg_response_length:800,avg_quality_score:.92,last_used:"2024-01-15T13:45:00Z",error_count:1,success_rate:99.5},{prompt_id:"efficient_summarization",execution_count:156,avg_execution_time:234,avg_response_length:450,avg_quality_score:.94,last_used:"2024-01-15T12:20:00Z",error_count:0,success_rate:100}],height:450}},n={args:{data:[{prompt_id:"slow_complex_analysis",execution_count:45,avg_execution_time:3456,avg_response_length:2100,avg_quality_score:.67,last_used:"2024-01-14T16:30:00Z",error_count:12,success_rate:73.3},{prompt_id:"unreliable_generator_v1",execution_count:78,avg_execution_time:2234,avg_response_length:890,avg_quality_score:.58,last_used:"2024-01-15T09:15:00Z",error_count:18,success_rate:76.9},{prompt_id:"legacy_prompt_old",execution_count:23,avg_execution_time:4567,avg_response_length:1500,avg_quality_score:.45,last_used:"2024-01-13T11:00:00Z",error_count:8,success_rate:65.2}],height:450}};var s,c,_;e.parameters={...e.parameters,docs:{...(s=e.parameters)==null?void 0:s.docs,source:{originalSource:`{
  args: {
    data: mockDataset.promptPerformance,
    height: 500
  }
}`,...(_=(c=e.parameters)==null?void 0:c.docs)==null?void 0:_.source}}};var i,u,m;t.parameters={...t.parameters,docs:{...(i=t.parameters)==null?void 0:i.docs,source:{originalSource:`{
  args: {
    data: generateMockPromptPerformance(25),
    height: 600
  }
}`,...(m=(u=t.parameters)==null?void 0:u.docs)==null?void 0:m.source}}};var p,g,l;r.parameters={...r.parameters,docs:{...(p=r.parameters)==null?void 0:p.docs,source:{originalSource:`{
  args: {
    data: generateMockPromptPerformance(5),
    height: 400
  }
}`,...(l=(g=r.parameters)==null?void 0:g.docs)==null?void 0:l.source}}};var d,h,v;a.parameters={...a.parameters,docs:{...(d=a.parameters)==null?void 0:d.docs,source:{originalSource:`{
  args: {
    data: mockDataset.promptPerformance.slice(0, 8),
    height: 400
  }
}`,...(v=(h=a.parameters)==null?void 0:h.docs)==null?void 0:v.source}}};var x,P,y;o.parameters={...o.parameters,docs:{...(x=o.parameters)==null?void 0:x.docs,source:{originalSource:`{
  args: {
    data: [{
      prompt_id: 'optimized_code_gen_v3',
      execution_count: 234,
      avg_execution_time: 156,
      avg_response_length: 1200,
      avg_quality_score: 0.95,
      last_used: '2024-01-15T14:30:00Z',
      error_count: 0,
      success_rate: 100
    }, {
      prompt_id: 'fast_text_analysis_v2',
      execution_count: 187,
      avg_execution_time: 89,
      avg_response_length: 800,
      avg_quality_score: 0.92,
      last_used: '2024-01-15T13:45:00Z',
      error_count: 1,
      success_rate: 99.5
    }, {
      prompt_id: 'efficient_summarization',
      execution_count: 156,
      avg_execution_time: 234,
      avg_response_length: 450,
      avg_quality_score: 0.94,
      last_used: '2024-01-15T12:20:00Z',
      error_count: 0,
      success_rate: 100
    }],
    height: 450
  }
}`,...(y=(P=o.parameters)==null?void 0:P.docs)==null?void 0:y.source}}};var f,b,q;n.parameters={...n.parameters,docs:{...(f=n.parameters)==null?void 0:f.docs,source:{originalSource:`{
  args: {
    data: [{
      prompt_id: 'slow_complex_analysis',
      execution_count: 45,
      avg_execution_time: 3456,
      avg_response_length: 2100,
      avg_quality_score: 0.67,
      last_used: '2024-01-14T16:30:00Z',
      error_count: 12,
      success_rate: 73.3
    }, {
      prompt_id: 'unreliable_generator_v1',
      execution_count: 78,
      avg_execution_time: 2234,
      avg_response_length: 890,
      avg_quality_score: 0.58,
      last_used: '2024-01-15T09:15:00Z',
      error_count: 18,
      success_rate: 76.9
    }, {
      prompt_id: 'legacy_prompt_old',
      execution_count: 23,
      avg_execution_time: 4567,
      avg_response_length: 1500,
      avg_quality_score: 0.45,
      last_used: '2024-01-13T11:00:00Z',
      error_count: 8,
      success_rate: 65.2
    }],
    height: 450
  }
}`,...(q=(b=n.parameters)==null?void 0:b.docs)==null?void 0:q.source}}};const G=["Default","ManyPrompts","FewPrompts","CompactView","HighPerformancePrompts","ProblematicPrompts"];export{a as CompactView,e as Default,r as FewPrompts,o as HighPerformancePrompts,t as ManyPrompts,n as ProblematicPrompts,G as __namedExportsOrder,A as default};
