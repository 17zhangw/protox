select min(w_warehouse_name)
  ,min(sm_type)
  ,min(cc_name)
  ,min(cs_order_number)
  ,min(cs_item_sk)
from
   catalog_sales
  ,warehouse
  ,ship_mode
  ,call_center
  ,date_dim
where
    d_month_seq between 1195 and 1195 + 23
and cs_ship_date_sk   = d_date_sk
and cs_warehouse_sk   = w_warehouse_sk
and cs_ship_mode_sk   = sm_ship_mode_sk
and cs_call_center_sk = cc_call_center_sk
and cs_list_price between 195 and 224
and sm_type = 'LIBRARY'
and cc_class = 'large'
and w_gmt_offset = -5;