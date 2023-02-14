select min(i_item_id),
        min(ca_country),
        min(ca_state),
        min(ca_county),
        min(cs_quantity),
        min(cs_list_price),
        min(cs_coupon_amt),
        min(cs_sales_price),
        min(cs_net_profit),
        min(c_birth_year),
        min(cd_dep_count)
 from catalog_sales, customer_demographics, customer, customer_address, date_dim, item
 where cs_sold_date_sk = d_date_sk and
       cs_item_sk = i_item_sk and
       cs_bill_cdemo_sk = cd_demo_sk and
       cs_bill_customer_sk = c_customer_sk and
       cd_gender = 'F' and
       cd_education_status = '4 yr Degree' and
       c_current_addr_sk = ca_address_sk and
       d_year = 1998 and
       c_birth_month = 12 and
       ca_state in ('CO', 'GA', 'OH')
       and cs_wholesale_cost BETWEEN 5 AND 10
       AND i_category = 'Sports';