/* SP3 same-width raw full-duplex spans. Included after tp4_transfer's definition.
 * One invocation owns one lane and both disjoint DDR regions until every
 * packet has completed. No CPU tensor copy or cache-maintenance loop occurs.
 */
#ifndef C209_EXECUTOR_DUPLEX_H
#define C209_EXECUTOR_DUPLEX_H
typedef struct {
 C206Lane lane;
 uint64_t tx_phys,rx_phys,dummy_tx_phys,dummy_rx_phys;
 uint32_t tx_bytes,rx_bytes,node,peer,lane_id,epoch,step,channel,timeout_ms;
} TP4DuplexSpanJob;

size_t cosmos_tp4_duplex_span_job_size(void){return sizeof(TP4DuplexSpanJob);}
uint32_t cosmos_tp4_duplex_kv_revision(void){return 1U;}

int cosmos_tp4_bulk_duplex_span(TP4DuplexSpanJob*j,C206StreamStats*s){
 if(!j||!s||!j->tx_bytes||!j->rx_bytes||
    j->tx_bytes>512U*1024U*1024U||j->rx_bytes>512U*1024U*1024U||
    j->tx_bytes%64||j->rx_bytes%64||!j->timeout_ms||j->lane_id!=2||
    !j->node||!j->peer||j->node==j->peer||j->node>TP4_SEQUENCE_PARALLEL||
    j->peer>TP4_SEQUENCE_PARALLEL||!tp4_direct_peer(j->node,j->peer,j->lane_id)||
    j->channel>255||!j->epoch||j->step>=TP4_UPDATES||
    !j->tx_phys||!j->rx_phys||!j->dummy_tx_phys||!j->dummy_rx_phys||
    j->tx_phys%64||j->rx_phys%64||j->dummy_tx_phys%64||j->dummy_rx_phys%64||
    j->tx_phys+j->tx_bytes<j->tx_phys||j->rx_phys+j->rx_bytes<j->rx_phys||
    (j->tx_phys<j->rx_phys+j->rx_bytes&&j->rx_phys<j->tx_phys+j->tx_bytes))return -5;
 *s=(C206StreamStats){0};uint64_t start=NOW(),deadline=start+(uint64_t)j->timeout_ms*1000000ULL;
 uint32_t offset=0,seq=0,previous_bytes=0;
 const uint32_t total=j->tx_bytes>j->rx_bytes?j->tx_bytes:j->rx_bytes;
 int error=0;
 while(offset<total){
  if(tp4_is_cancelled()){error=-15;break;}
  uint64_t now=NOW();if(now>=deadline){error=-1;break;}
  uint64_t remaining=(deadline-now)/1000000ULL;if(!remaining){error=-1;break;}
  uint32_t tx_left=offset<j->tx_bytes?j->tx_bytes-offset:0;
  uint32_t rx_left=offset<j->rx_bytes?j->rx_bytes-offset:0;
  uint32_t n=total-offset;if(n>TP4_PACKET_MAX_BYTES)n=TP4_PACKET_MAX_BYTES;
  /* Split at either real span boundary, not just the longest boundary.
   * This keeps every real access in-bounds even when VS lengths are5632
   * versus5120bytes. The peer computes the same n with TX/RX swapped.
   * Only an exhausted direction uses the existing8192byte lane scratch.
   */
  if(tx_left&&n>tx_left)n=tx_left;
  if(rx_left&&n>rx_left)n=rx_left;
  uint64_t tx=tx_left?j->tx_phys+offset:j->dummy_tx_phys;
  uint64_t rx=rx_left?j->rx_phys+offset:j->dummy_rx_phys;
  error=tp4_transfer(&j->lane,tx,rx,0,n,j->node,j->peer,2,0x80,
    j->epoch,j->step,0x80,0,j->channel,seq,(uint32_t)remaining,2,
    seq!=0&&n==previous_bytes);
  if(error)break;
  previous_bytes=n;offset+=n;seq++;
  s->chunks++;s->tx_bytes+=n;s->rx_bytes+=n;
 }
 s->elapsed_ns=NOW()-start;s->error=error?(uint32_t)-error:0;return error;
}
#endif
