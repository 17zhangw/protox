with customer_total_return as
(select sr_customer_sk as ctr_customer_sk
,sr_store_sk as ctr_store_sk
,sr_reason_sk as ctr_reason_sk
,sum(SR_RETURN_AMT) as ctr_total_return
from store_returns
,date_dim
where sr_returned_date_sk = d_date_sk
and d_year =1999
and sr_return_amt / sr_return_quantity between 175 and 234
group by sr_customer_sk
,sr_store_sk, sr_reason_sk)
 select  c_customer_id
from customer_total_return ctr1
,store
,customer
,customer_demographics
where ctr1.ctr_total_return > (select avg(ctr_total_return)*1.2
from customer_total_return ctr2
where ctr1.ctr_store_sk = ctr2.ctr_store_sk
)
and ctr1.ctr_reason_sk BETWEEN 18 AND 21
and s_store_sk = ctr1.ctr_store_sk
and s_state IN ('AR', 'VT', 'WI')
and ctr1.ctr_customer_sk = c_customer_sk
and c_current_cdemo_sk = cd_demo_sk
and cd_marital_status IN ('S', 'D')
and cd_education_status IN ('Primary', 'Secondary')
and cd_gender = 'M'
and c_birth_month = 7
and c_birth_year BETWEEN 1940 AND 1946
order by c_customer_id
limit 100;