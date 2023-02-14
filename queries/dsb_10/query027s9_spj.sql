select min(i_item_id),
        min(s_state),
        min(ss_quantity),
        min(ss_list_price),
        min(ss_coupon_amt),
        min(ss_sales_price),
        min(ss_item_sk),
        min(ss_ticket_number)
 from store_sales, customer_demographics, date_dim, store, item
 where ss_sold_date_sk = d_date_sk and
       ss_item_sk = i_item_sk and
       ss_store_sk = s_store_sk and
       ss_cdemo_sk = cd_demo_sk and
       cd_gender = 'M' and
       cd_marital_status = 'W' and
       cd_education_status = 'Unknown' and
       d_year = 2002 and
       s_state = 'MI' and
      i_category  = 'Children';