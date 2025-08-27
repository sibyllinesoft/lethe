import{T as M}from"./TimelineChart-6880282a.js";import{m as s,b as y}from"./mockData-38ebf9b0.js";import"./createLucideIcon-a1714018.js";import"./index-8b3efc3f.js";import"./_commonjsHelpers-de833af9.js";import"./charts-bbe52c79.js";import"./generateCategoricalChart-ec1bcc95.js";import"./throttle-32ea0a72.js";import"./mapValues-fb986170.js";import"./tiny-invariant-dd7d57d2.js";import"./isPlainObject-cba39321.js";import"./_baseUniq-e6e71c30.js";import"./Line-c0585f89.js";const G={title:"Components/TimelineChart",component:M,parameters:{layout:"padded",docs:{description:{component:"Interactive timeline chart showing execution metrics over time with line visualizations for executions, response time, and errors."}}},tags:["autodocs"],argTypes:{height:{control:{type:"range",min:300,max:800,step:50}}}},e={args:{data:s.timelineData,height:400}},t={args:{data:y(7),height:400}},a={args:{data:y(30),height:500}},n={args:{data:s.timelineData,height:300}},r={args:{data:s.timelineData,height:600}},o={args:{data:[{date:"2024-01-10",total_executions:450,avg_execution_time:234,avg_response_length:1200,errors:2},{date:"2024-01-11",total_executions:523,avg_execution_time:198,avg_response_length:1100,errors:1},{date:"2024-01-12",total_executions:612,avg_execution_time:167,avg_response_length:1250,errors:0},{date:"2024-01-13",total_executions:578,avg_execution_time:189,avg_response_length:1180,errors:3},{date:"2024-01-14",total_executions:634,avg_execution_time:156,avg_response_length:1300,errors:1},{date:"2024-01-15",total_executions:689,avg_execution_time:145,avg_response_length:1350,errors:0},{date:"2024-01-16",total_executions:701,avg_execution_time:134,avg_response_length:1400,errors:2}],height:450}};var i,c,g;e.parameters={...e.parameters,docs:{...(i=e.parameters)==null?void 0:i.docs,source:{originalSource:`{
  args: {
    data: mockDataset.timelineData,
    height: 400
  }
}`,...(g=(c=e.parameters)==null?void 0:c.docs)==null?void 0:g.source}}};var m,_,p;t.parameters={...t.parameters,docs:{...(m=t.parameters)==null?void 0:m.docs,source:{originalSource:`{
  args: {
    data: generateMockTimelineData(7),
    height: 400
  }
}`,...(p=(_=t.parameters)==null?void 0:_.docs)==null?void 0:p.source}}};var l,d,u;a.parameters={...a.parameters,docs:{...(l=a.parameters)==null?void 0:l.docs,source:{originalSource:`{
  args: {
    data: generateMockTimelineData(30),
    height: 500
  }
}`,...(u=(d=a.parameters)==null?void 0:d.docs)==null?void 0:u.source}}};var h,v,x;n.parameters={...n.parameters,docs:{...(h=n.parameters)==null?void 0:h.docs,source:{originalSource:`{
  args: {
    data: mockDataset.timelineData,
    height: 300
  }
}`,...(x=(v=n.parameters)==null?void 0:v.docs)==null?void 0:x.source}}};var D,k,T;r.parameters={...r.parameters,docs:{...(D=r.parameters)==null?void 0:D.docs,source:{originalSource:`{
  args: {
    data: mockDataset.timelineData,
    height: 600
  }
}`,...(T=(k=r.parameters)==null?void 0:k.docs)==null?void 0:T.source}}};var C,f,S;o.parameters={...o.parameters,docs:{...(C=o.parameters)==null?void 0:C.docs,source:{originalSource:`{
  args: {
    data: [{
      date: '2024-01-10',
      total_executions: 450,
      avg_execution_time: 234,
      avg_response_length: 1200,
      errors: 2
    }, {
      date: '2024-01-11',
      total_executions: 523,
      avg_execution_time: 198,
      avg_response_length: 1100,
      errors: 1
    }, {
      date: '2024-01-12',
      total_executions: 612,
      avg_execution_time: 167,
      avg_response_length: 1250,
      errors: 0
    }, {
      date: '2024-01-13',
      total_executions: 578,
      avg_execution_time: 189,
      avg_response_length: 1180,
      errors: 3
    }, {
      date: '2024-01-14',
      total_executions: 634,
      avg_execution_time: 156,
      avg_response_length: 1300,
      errors: 1
    }, {
      date: '2024-01-15',
      total_executions: 689,
      avg_execution_time: 145,
      avg_response_length: 1350,
      errors: 0
    }, {
      date: '2024-01-16',
      total_executions: 701,
      avg_execution_time: 134,
      avg_response_length: 1400,
      errors: 2
    }],
    height: 450
  }
}`,...(S=(f=o.parameters)==null?void 0:f.docs)==null?void 0:S.source}}};const J=["Default","OneWeek","OneMonth","CompactHeight","TallChart","HighActivity"];export{n as CompactHeight,e as Default,o as HighActivity,a as OneMonth,t as OneWeek,r as TallChart,J as __namedExportsOrder,G as default};
