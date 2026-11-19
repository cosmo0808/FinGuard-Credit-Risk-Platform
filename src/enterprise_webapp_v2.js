// Stage 2 workflow: empirical short-circuit branches and transparent review sensitivity.
function renderWorkflowV2(){
  const w=D.workflow, rules=Object.fromEntries(w.rules.map(x=>[x.rule_id,x])),r01=rules.R01,r02=rules.R02;
  const blocked=r01.hits+r02.hits,eligible=w.applications-blocked,approved=w.models['Challenger LightGBM M4'].approved,declined=eligible-approved;
  const badPct=fmt(w.base_bad_rate_pct,2);
  $('#workflowDag').innerHTML=`<div class="branch-intake" data-applications="${w.applications}"><b>Intake</b><strong>${fmt(w.applications,0)}</strong><small>${badPct}% bad · OOT_Monitor</small></div><div class="branch-row"><section class="branch-node blocked" data-count="${blocked}"><b>Hard-Rule Blocked</b><strong>${fmt(blocked,0)}</strong><small>R01 ${fmt(r01.hits,0)} · R02 net ${fmt(r02.hits,0)}</small><small>Short-circuit path; excluded from model approval stage</small></section><section class="branch-node eligible" data-count="${eligible}"><b>Rule-Eligible Scored</b><strong>${fmt(eligible,0)}</strong><small>Intake − R01 − net R02</small><div class="branch-outcomes"><div class="branch-node declined" data-count="${declined}"><b>Model Declined</b><strong>${fmt(declined,0)}</strong><small>Challenger Iso-Risk</small></div><div class="branch-node approved" data-count="${approved}"><b>Approved & Limit-Assigned</b><strong>${fmt(approved,0)}</strong><small>${fmt(100*approved/w.applications,2)}% of total · ${fmt(w.models['Challenger LightGBM M4'].bad_rate_pct,4)}% bad</small></div></div></section></div>`;
  $('#ruleWaterfall').innerHTML=table(w.rules,[['rule_id','ID'],['rule','Rule'],['hits','Hits',v=>fmt(v,0)],['hit_pct','Hit %',v=>fmt(v,2)+'%'],['bads','Bads',v=>fmt(v,0)],['bad_rate_pct','Bad rate %',v=>v==null?'—':fmt(v,3)+'%'],['bad_capture_lift','Bad-capture lift',v=>v==null?'—':fmt(v,2)+'×']]);
  $('#ruleWaterfall table').classList.add('workflow-table');
  const insight=document.createElement('p');insight.className='notice';insight.textContent=`MRM insight: R02 is an affordability ceiling with minimal standalone default lift (${fmt(r02.bad_capture_lift,2)}×); interpret it as capacity protection, not a high-risk default detector.`;$('#ruleWaterfall').after(insight);
  const t=w.tri_state,exposure=Number(t.review_exposure_mean||0),queue=Number(t.review_count||0),badRate=Number(t.review_bad_rate_pct||0)/100,lgd=.75,cost=25;
  const sensitivity=[0,.05,.10,.20].map(cure=>{const avoided=queue*badRate*cure,impact=avoided*exposure*lgd-queue*cost;return {'Assumed Cure Rate':fmt(cure*100,0)+'%','Review Cases':queue,'Expected Avoided Bads':avoided,'Net Contribution Impact':impact}});
  $('#reviewMetrics').innerHTML=table(sensitivity,[['Assumed Cure Rate','Assumed Cure Rate'],['Review Cases','Review Cases',v=>fmt(v,0)],['Expected Avoided Bads','Avoided bads',v=>fmt(v,2)],['Net Contribution Impact','Net contribution impact',v=>fmt(v,2)]]);
  const apr=['9%','12%','15%','18%','Auto-decline'];
  const mult=['0.80x','0.75x','0.70x','0.60x','0.00x / Auto-decline'];
  const matrix=w.risk_matrix.map((r,i)=>({'Risk Tier':`Tier ${String.fromCharCode(65+i)}`,'Illustrative APR Band':apr[i],'Verified Income Limit Rule':'min(request, risk/behavior cap, 50% total-DSR annuity PV)','Unverified Income Fallback Rule':mult[i]}));
  $('#riskMatrix').innerHTML=table(matrix,[['Risk Tier','Risk Tier'],['Illustrative APR Band','Illustrative APR Band'],['Verified Income Limit Rule','Verified income rule'],['Unverified Income Fallback Rule','Unverified income fallback']]);$('#riskMatrix table').classList.add('workflow-table');
}
renderWorkflowV2();
