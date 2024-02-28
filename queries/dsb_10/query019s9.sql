select  i_brand_id brand_id, i_brand brand, i_manufact_id, i_manufact,
 	sum(ss_ext_sales_price) ext_price
  from date_dim, store_sales, item,customer,customer_address,store
  where  d_date_sk = ss_sold_date_sk
    and ss_item_sk = i_item_sk
    and ss_customer_sk = c_customer_sk
    and c_current_addr_sk = ca_address_sk
    and ss_store_sk = s_store_sk
    AND i_category  = 'Books'
    and d_year=2002
    and d_moy = 12
    and substring(ca_zip,1,5) <> substring(s_zip,1,5)
    and ca_state  = 'IA'
    and c_birth_month = 7
    and ss_wholesale_cost BETWEEN 9 AND 29
 group by i_brand
      ,i_brand_id
      ,i_manufact_id
      ,i_manufact
 order by ext_price desc
         ,i_brand
         ,i_brand_id
         ,i_manufact_id
         ,i_manufact
limit 100;