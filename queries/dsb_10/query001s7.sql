with customer_total_return as
(select sr_customer_sk as ctr_customer_sk
,sr_store_sk as ctr_store_sk
,sr_reason_sk as ctr_reason_sk
,sum(SR_RETURN_TAX) as ctr_total_return
from store_returns
,date_dim
where sr_returned_date_sk = d_date_sk
and d_year =2002
and sr_return_amt / sr_return_quantity between 179 and 238
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
and ctr1.ctr_reason_sk BETWEEN 13 AND 16
and s_store_sk = ctr1.ctr_store_sk
and s_state IN ('KS', 'MS', 'TX')
and ctr1.ctr_customer_sk = c_customer_sk
and c_current_cdemo_sk = cd_demo_sk
and cd_marital_status IN ('M', 'U')
and cd_education_status IN ('Unknown', '2 yr Degree')
and cd_gender = 'M'
and c_birth_month = 9
and c_birth_year BETWEEN 1927 AND 1933
order by c_customer_id
limit 100;