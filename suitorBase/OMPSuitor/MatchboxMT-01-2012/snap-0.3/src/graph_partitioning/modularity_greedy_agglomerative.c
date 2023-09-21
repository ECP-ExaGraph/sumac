#include <string.h>
#include "graph_partitioning.h"
#include "graph_kernels.h"
#include "graph_kernels.h"

/* Community adjacency structure -- sorted array */
typedef struct s_commpair_aggc_t {
    attr_id_t comm_id; /* the list is sorted by this value */  
    double dq;           /* increase in modularity for this pair */
} commpair_aggc_t;

/* Structure representing a community and its adjacencies */
typedef struct {
    commpair_aggc_t* adjcomm; /* communities adjacent to current comm */
    attr_id_t degree;         /* number of adjacent communities */
    attr_id_t max_dq_idx;     /* neighbor with max dq value */
    double max_dq;            /* max dq value */
    double a;                 /* a, sum of degrees of all vertices in community */
    attr_id_t max_size;       /* max size of the adjcomm array, array needs to 
                                 be resized id degree = max_size */
    attr_id_t comm_size;      /* no. of vertices that belong to a community */
    attr_id_t parent_id;      /* parent of a vertex (set once community is 
                                 merged) */
    attr_id_t resized;        /* flag to indicate if a community adjacency array 
                                 has been resized after creation. Set to 1 on 
                                 resize */
    /* char[20] filler; */    /* can align it to cache line size */
} aggc_comm_t;

/* Implicit max heap struct for storing the modularity 
 * delta values for community pairs */
typedef struct {
    commpair_aggc_t* heap;
    attr_id_t* index;
    attr_id_t n;
} aggc_maxheap_t;

int comp_comm(const void *v1, const void *v2) {
    const commpair_aggc_t *p1, *p2;
    p1 = v1;
    p2 = v2;

    if (p1->comm_id < p2->comm_id)
        return -1;
    else if (p1->comm_id == p2->comm_id)
        return 0;
    else
        return 1;
}

void aggc_maxheap_sift_up(aggc_maxheap_t *dq_maxheap, attr_id_t idx) {

    commpair_aggc_t *comm_list;
    attr_id_t *heap_pos;
    attr_id_t root, parent, u, v, n, pos;
    double tmp;

    comm_list = dq_maxheap->heap;
    heap_pos  = dq_maxheap->index;
    n         = dq_maxheap->n;

    root = idx;
    while (root > 0) {
        parent = (root-1)/2;
        if (comm_list[parent].dq < comm_list[root].dq) {
            /* swap */
            tmp = comm_list[root].dq;
            v  = comm_list[root].comm_id;
            u  = comm_list[parent].comm_id;

            comm_list[root].dq = comm_list[parent].dq;
            comm_list[root].comm_id = u;
            
            comm_list[parent].dq = tmp;
            comm_list[parent].comm_id = v;
       
            pos = heap_pos[v];
            heap_pos[v] = heap_pos[u];
            heap_pos[u] = pos;

            root = parent;

        } else break;
    }
}

void aggc_maxheap_sift_down(aggc_maxheap_t *dq_maxheap, attr_id_t idx) {

    commpair_aggc_t *comm_list;
    attr_id_t *heap_pos;
    attr_id_t root, child, u, v, n, pos;
    double tmp;

    comm_list = dq_maxheap->heap;
    heap_pos  = dq_maxheap->index;
    n         = dq_maxheap->n;

    root = idx;
    while (root*2+1 < n) {
        child = root*2+1;
        if (child+1 < n) 
           if (comm_list[child].dq < comm_list[child+1].dq)
              child++;
        if (comm_list[root].dq < comm_list[child].dq) {
            /* swap */
            tmp = comm_list[root].dq;
            v  = comm_list[root].comm_id;
            u  = comm_list[child].comm_id;

            comm_list[root].dq = comm_list[child].dq;
            comm_list[root].comm_id = u;
            
            comm_list[child].dq = tmp;
            comm_list[child].comm_id = v;
       
            pos = heap_pos[v];
            heap_pos[v] = heap_pos[u];
            heap_pos[u] = pos;

            root = child;

        } else break;
    }
}

void aggc_maxheap_remove(aggc_maxheap_t* dq_maxheap, attr_id_t comm_id) {
    commpair_aggc_t *comm_list;
    attr_id_t *heap_pos;
    attr_id_t idx, new_comm_id, n;
    double old_dq;

    comm_list = dq_maxheap->heap;
    heap_pos  = dq_maxheap->index;
    n         = dq_maxheap->n;

    idx = heap_pos[comm_id];
    if (idx < 0)
        return;
    assert(idx != -1);
    old_dq = comm_list[idx].dq;

    /* Move last value to this position */
    new_comm_id = comm_list[n-1].comm_id;
    comm_list[idx].comm_id = new_comm_id;
    comm_list[idx].dq = comm_list[n-1].dq;
   
    heap_pos[new_comm_id] = idx;
    heap_pos[comm_id] = -1;
    
    dq_maxheap->n = n-1;

    if (old_dq > comm_list[idx].dq) {
        aggc_maxheap_sift_down(dq_maxheap, idx);
    } else {
        aggc_maxheap_sift_up(dq_maxheap, idx);
    }
}

void aggc_maxheap_check(aggc_maxheap_t* dq_maxheap) {
    attr_id_t i, n;
    commpair_aggc_t* comm_list;
    
    comm_list = dq_maxheap->heap;
    n         = dq_maxheap->n;

    for (i=0; i<n/2; i++) {
        if ((2*i+1<n && comm_list[i].dq < comm_list[2*i+1].dq) 
                || (2*i+2<n && comm_list[i].dq < comm_list[2*i+2].dq)) {
            fprintf(stderr, "heap property violated\n");
        }
    }
}

attr_id_t aggc_adjcomm_pos(commpair_aggc_t *adjcomm, attr_id_t val, 
        attr_id_t degree) {

    attr_id_t low, high, mid;

    low = 0;
    high = degree-1;
    while (low < high) {
        mid = low + ((high-low)/2);
        if (adjcomm[mid].comm_id < val)
            low = mid + 1;
        else 
            high = mid;
    }

    return low;

}

void aggc_adjcomm_resize(aggc_comm_t *communities, attr_id_t comm_id, 
        attr_id_t incr) {

    attr_id_t new_size;
    commpair_aggc_t *new_adjcomm, *old_adjcomm;

    old_adjcomm = communities[comm_id].adjcomm;
    new_size = communities[comm_id].max_size + 2*incr;

    communities[comm_id].max_size = new_size;

    new_adjcomm = (commpair_aggc_t *) malloc(new_size * 
            sizeof(commpair_aggc_t));
    assert(new_adjcomm != NULL);
    memcpy(new_adjcomm, old_adjcomm, 
            communities[comm_id].degree * sizeof(commpair_aggc_t));
    if (communities[comm_id].resized == 1) {
        free(old_adjcomm);
    } else {
        communities[comm_id].resized = 1;
    }
    communities[comm_id].adjcomm = new_adjcomm;
}

void aggc_update_dq_p1(aggc_comm_t *communities, aggc_maxheap_t* dq_maxheap,
        attr_id_t comm1, attr_id_t comm2, double new_dq_val) {

    attr_id_t i, max_dq_idx, max_dq_commpair_id, pos, degree;
    commpair_aggc_t *adjcomm, *dq_comm_list;
    attr_id_t *dq_index;
    double max_dq, max_dq_old, dq_val;
    attr_id_t heap_pos;
    double INFTY;
    
    INFTY = 100000.0;
    dq_comm_list = dq_maxheap->heap;
    dq_index     = dq_maxheap->index;

    max_dq_idx = communities[comm1].max_dq_idx;
    max_dq     = max_dq_old = communities[comm1].max_dq;
    
    heap_pos = dq_index[comm1];
    assert(heap_pos != -1);

    assert(max_dq_idx >= 0);
    max_dq_commpair_id = communities[comm1].adjcomm[max_dq_idx].comm_id;
    degree = communities[comm1].degree;
    adjcomm = communities[comm1].adjcomm;

    if (comm2 == max_dq_commpair_id) {
        adjcomm[max_dq_idx].dq = new_dq_val;
        if (new_dq_val > max_dq) {
            max_dq = new_dq_val;
            communities[comm1].max_dq = new_dq_val;
        } else {
            max_dq = -INFTY;
            max_dq_idx = -1;
            /* Rescan adjacencies to determine largest dq value */
            for (i=0; i<degree; i++) {
                dq_val = adjcomm[i].dq;
                if (dq_val > max_dq) {
                    max_dq_idx = i;
                    max_dq     = adjcomm[i].dq;
                }
            }
            communities[comm1].max_dq = max_dq;
            communities[comm1].max_dq_idx = max_dq_idx;
        }   
        /* Sift up/down value in max heap */
        heap_pos = dq_index[comm1];
        dq_comm_list[heap_pos].dq = max_dq;
        if (max_dq > max_dq_old) 
            aggc_maxheap_sift_up(dq_maxheap, heap_pos);
        else
            aggc_maxheap_sift_down(dq_maxheap, heap_pos);

    } else {
        /* Locate this adjacency */
        pos = aggc_adjcomm_pos(adjcomm, comm2, degree);

        /* Update its dq value */
        adjcomm[pos].dq = new_dq_val;
        if (new_dq_val > max_dq) {
            /* Update global adj value */
            communities[comm1].max_dq = new_dq_val;
            communities[comm1].max_dq_idx = pos;

            /* Sift up/down value in max heap */
            dq_comm_list[heap_pos].dq = new_dq_val;
            aggc_maxheap_sift_up(dq_maxheap, heap_pos);
        }
    }
}
 
void aggc_update_dq_p2(aggc_comm_t *communities, aggc_maxheap_t* dq_maxheap,
        attr_id_t comm1, attr_id_t comm2, attr_id_t new_comm_id, 
        double new_dq_val) {

    attr_id_t i, max_dq_idx, max_dq_commpair_id, pos, degree;
    commpair_aggc_t *adjcomm, *dq_comm_list;
    attr_id_t *dq_index, heap_pos, INFTY;
    double max_dq, dq_val, max_dq_old;

    INFTY = 100000.0;
    dq_comm_list = dq_maxheap->heap;
    dq_index     = dq_maxheap->index;

    max_dq_idx = communities[comm1].max_dq_idx;
    assert(max_dq_idx >= 0);
    max_dq     = max_dq_old = communities[comm1].max_dq;
    max_dq_commpair_id = communities[comm1].adjcomm[max_dq_idx].comm_id;
    degree = communities[comm1].degree;
    adjcomm = communities[comm1].adjcomm;

    if (comm2 == max_dq_commpair_id) {
        adjcomm[max_dq_idx].dq = new_dq_val;
        adjcomm[max_dq_idx].comm_id = new_comm_id;

        if (new_dq_val > max_dq) {
            max_dq = new_dq_val;
            communities[comm1].max_dq = new_dq_val;
            
            /* move this adjacency to correct position */
            if (new_comm_id > comm2) {
                for (i=max_dq_idx; i<degree-1; i++) {
                    if (new_comm_id > adjcomm[i+1].comm_id) {
                        /* swap */
                        adjcomm[i].comm_id = adjcomm[i+1].comm_id;
                        adjcomm[i].dq      = adjcomm[i+1].dq;
                        adjcomm[i+1].comm_id = new_comm_id;
                        adjcomm[i+1].dq      = max_dq;
                    } else {
                        break;
                    }
                }
                communities[comm1].max_dq_idx = i;
            } else {
                for (i=max_dq_idx; i>0; i--) {
                    if (new_comm_id < adjcomm[i-1].comm_id) {
                        /* swap */
                        adjcomm[i].comm_id = adjcomm[i-1].comm_id;
                        adjcomm[i].dq      = adjcomm[i-1].dq;
                        adjcomm[i-1].comm_id = new_comm_id;
                        adjcomm[i-1].dq  = max_dq;
                    } else {
                        break;
                    }
                }
                communities[comm1].max_dq_idx = i;
            }

        } else {

            /* move this adjacency to correct position */
            if (new_comm_id > comm2) {
                for (i=max_dq_idx; i<degree-1; i++) {
                    if (new_comm_id > adjcomm[i+1].comm_id) {
                        /* swap */
                        adjcomm[i].comm_id = adjcomm[i+1].comm_id;
                        adjcomm[i].dq      = adjcomm[i+1].dq;
                        adjcomm[i+1].comm_id = new_comm_id;
                        adjcomm[i+1].dq      = new_dq_val;
                    } else {
                        break;
                    }
                }
            } else {
                for (i=max_dq_idx; i>0; i--) {
                    if (new_comm_id < adjcomm[i-1].comm_id) {
                        /* swap */
                        adjcomm[i].comm_id = adjcomm[i-1].comm_id;
                        adjcomm[i].dq      = adjcomm[i-1].dq;
                        adjcomm[i-1].comm_id = new_comm_id;
                        adjcomm[i-1].dq  = new_dq_val;
                    } else {
                        break;
                    }
                }
            }
            max_dq_idx = -1;
            max_dq = -INFTY;
            /* Rescan adjacencies to determine largest dq value */
            for (i=0; i<degree; i++) {
                dq_val = adjcomm[i].dq;
                if (dq_val > max_dq) {
                    max_dq_idx = i;
                    max_dq     = adjcomm[i].dq;
                }
            }
            communities[comm1].max_dq = max_dq;
            communities[comm1].max_dq_idx = max_dq_idx;
        }   
        
        /* Sift up/down value in max heap */
        heap_pos = dq_index[comm1];
        assert(heap_pos != -1);

        dq_comm_list[heap_pos].dq = max_dq;
        if (max_dq > max_dq_old) 
            aggc_maxheap_sift_up(dq_maxheap, heap_pos);
        else
            aggc_maxheap_sift_down(dq_maxheap, heap_pos);

    } else {

        /* Locate this adjacency */
        pos = aggc_adjcomm_pos(adjcomm, comm2, degree);
        adjcomm[pos].dq = new_dq_val; 
        adjcomm[pos].comm_id = new_comm_id;
        /* move this adjacency to correct position */
        if (new_comm_id > comm2) {
            for (i=pos; i<degree-1; i++) {
                if (new_comm_id > adjcomm[i+1].comm_id) {
                    /* swap */
                    adjcomm[i].comm_id = adjcomm[i+1].comm_id;
                    adjcomm[i].dq      = adjcomm[i+1].dq;
                    adjcomm[i+1].comm_id = new_comm_id;
                    adjcomm[i+1].dq      = new_dq_val;
                } else {
                    break;
                }
            }
        } else {
            for (i=pos; i>0; i--) {
                if (new_comm_id < adjcomm[i-1].comm_id) {
                    /* swap */
                    adjcomm[i].comm_id = adjcomm[i-1].comm_id;
                    adjcomm[i].dq      = adjcomm[i-1].dq;
                    adjcomm[i-1].comm_id = new_comm_id;
                    adjcomm[i-1].dq  = new_dq_val;
                } else {
                    break;
                }
            }
        }

        /* Update its dq value */
        if (new_dq_val > max_dq) {
            /* Update global adj value */
            communities[comm1].max_dq = new_dq_val;
            communities[comm1].max_dq_idx = i;

            /* Sift up/down value in max heap */
            heap_pos = dq_index[comm1];
            dq_comm_list[heap_pos].dq = new_dq_val;
            aggc_maxheap_sift_up(dq_maxheap, heap_pos);
        } else {
            /* update local max_dq_idx */
            max_dq_idx = communities[comm1].max_dq_idx;
            if (new_comm_id > comm2) {
                if ((max_dq_idx > pos) && (max_dq_idx <= i)) {
                    communities[comm1].max_dq_idx = max_dq_idx - 1;
                }
            } else {
                if ((max_dq_idx >= i) && (max_dq_idx < pos)) {
                    communities[comm1].max_dq_idx = max_dq_idx + 1;
                }
            }
        }
    }
}
 
              
void aggc_remove_commpair(aggc_comm_t *communities, aggc_maxheap_t *dq_maxheap,
        attr_id_t comm1, attr_id_t comm2) {

    attr_id_t i, comm_id, max_dq_idx, max_dq_commpair_id, pos, degree;
    commpair_aggc_t *adjcomm, *dq_comm_list;
    double max_dq, max_dq_old, dq_val;
    attr_id_t *dq_index;
    attr_id_t heap_pos;
    double INFTY;

    INFTY = 100000.0;
    dq_index = dq_maxheap->index;
    dq_comm_list = dq_maxheap->heap;

    max_dq_idx = communities[comm1].max_dq_idx;
    max_dq_old = communities[comm1].max_dq;

    max_dq_commpair_id = communities[comm1].adjcomm[max_dq_idx].comm_id;
    degree = communities[comm1].degree;
    adjcomm = communities[comm1].adjcomm;

    if (comm2 == max_dq_commpair_id) {
                
        /* we need to rescan to find new max_dq for comm1 */
        max_dq = -INFTY;
        max_dq_idx = -1;
        communities[comm1].max_dq = -INFTY;
        communities[comm1].max_dq_idx = -1;

        i = 0;
        while (i < degree) {
            comm_id = adjcomm[i].comm_id;
            dq_val = adjcomm[i].dq;
            if (comm_id == comm2)
                break;
            if (dq_val > max_dq) {
                max_dq = dq_val;
                max_dq_idx = i;
            }
            i++;
        }
        
        pos = i;
        for (i=pos; i<degree-1; i++) {
            adjcomm[i].comm_id = adjcomm[i+1].comm_id;
            adjcomm[i].dq  = adjcomm[i+1].dq;

            if (adjcomm[i].dq > max_dq) {
                max_dq = adjcomm[i].dq;
                max_dq_idx = i;
            }
        }

       communities[comm1].max_dq = max_dq;
       communities[comm1].max_dq_idx = max_dq_idx; 
  
       /* Sift down value in max-heap */
       heap_pos = dq_index[comm1];
       dq_comm_list[heap_pos].dq = max_dq;
       if (max_dq > max_dq_old) 
           aggc_maxheap_sift_up(dq_maxheap, heap_pos);
       else
           aggc_maxheap_sift_down(dq_maxheap, heap_pos);

    } else {

        /* do a binary search to locate the pair in adjcomm, delete it,
         * and update the communities ordering */
        pos = aggc_adjcomm_pos(adjcomm, comm2, degree);
        if (max_dq_idx > pos)
            communities[comm1].max_dq_idx--;

        for (i=pos; i<degree-1; i++) {
            adjcomm[i].comm_id = adjcomm[i+1].comm_id;
            adjcomm[i].dq  = adjcomm[i+1].dq;
        }
    }

    communities[comm1].degree--;

}
         
void aggc_merge_communities_cnm(aggc_comm_t* communities, 
        aggc_maxheap_t* dq_maxheap, 
        double *dq_increase_ptr, commpair_aggc_t* adj_buffer) {

        double INFTY;
        attr_id_t HIGH_VID;
        attr_id_t p1, p2, comm1, comm2, from, to, i, j, from_size, to_size;
        commpair_aggc_t *from_adj, *to_adj, *dq_comm_list;
        double max_dq, from_a, to_a, max_dq_old;
        attr_id_t ins_pos, heap_pos, max_dq_idx, max_dq_idx_comm2, curr_i_idx;
        attr_id_t new_neis;
        attr_id_t *dq_index;

        dq_index = dq_maxheap->index;
        dq_comm_list = dq_maxheap->heap;

        INFTY = 100000.0;
        HIGH_VID = 1<<30;
        comm1 = dq_comm_list[0].comm_id;
        max_dq = max_dq_old = dq_comm_list[0].dq;

        *dq_increase_ptr = max_dq;

        if (max_dq < 0) {
            return;
        }

        max_dq_idx = communities[comm1].max_dq_idx;
        comm2 = communities[comm1].adjcomm[max_dq_idx].comm_id;
 
        max_dq_idx_comm2 = communities[comm2].max_dq_idx;
        
        if (comm1 != communities[comm2].adjcomm[max_dq_idx_comm2].comm_id) {
            /* comm2 has an adjacency comm3 which has the same dq value 
               as the pair comm1 -> comm2 */
           /* Search to locate comm1 and modify comm2's max_dq_idx */
          ins_pos = aggc_adjcomm_pos(communities[comm2].adjcomm, comm1, 
                    communities[comm2].degree);
          communities[comm2].max_dq_idx = ins_pos;
        }

        if (communities[comm1].degree < communities[comm2].degree) {
            from = comm1;
            to = comm2;
        } else {
            from = comm2;
            to = comm1;
        }

        i = j = 0;
    
        from_size = communities[from].degree;
        from_adj  = communities[from].adjcomm;
        from_a    = communities[from].a;
        to_size   = communities[to].degree;
        to_adj    = communities[to].adjcomm;
        to_a      = communities[to].a;

        max_dq = -INFTY;
        max_dq_idx = -1;
        curr_i_idx = 0;
        new_neis = 0;

        /* fprintf(stderr, "Merge communities %d -> %d, max_dq %d\n", from, to, 
                max_dq_old);
         */
        while (i<to_size && j<from_size) {
            
            p1 = to_adj[i].comm_id;
            p2 = from_adj[j].comm_id;

            if (p1 == from) {
                /* This community needs to be deleted */
                to_adj[i].comm_id = HIGH_VID;
                i++;
                continue;
            }

            if (p2 == to) {
                j++;
                continue;
            }

            if (p1 < p2) {
                /* We have a community pair that is in "to" 
                 * but not in "from" */
                /* Update its dq value */
                to_adj[i].dq -= 2*from_a*communities[p1].a;
                if (to_adj[i].dq > max_dq) {
                    max_dq = to_adj[i].dq;
                    max_dq_idx = curr_i_idx;
                }
                /* Update dq value of commpair (p1 -> to) */
                aggc_update_dq_p1(communities, dq_maxheap, p1, to, to_adj[i].dq);
                i++;
                curr_i_idx++;
                continue;
            }

            if (p1 == p2) {
                to_adj[i].dq += from_adj[j].dq;
                if (to_adj[i].dq > max_dq) {
                    max_dq = to_adj[i].dq;
                    max_dq_idx = curr_i_idx;
                }
                aggc_update_dq_p1(communities, dq_maxheap, p1, to, to_adj[i].dq);
                aggc_remove_commpair(communities, dq_maxheap, p2, from);
               
                i++;
                j++;
                curr_i_idx++;
                continue;
            }

            if (p1 > p2) {
                if (communities[to].degree == communities[to].max_size) {
                    aggc_adjcomm_resize(communities, to, from_size);
                    to_adj = communities[to].adjcomm;
                } 
                ins_pos = communities[to].degree;
                communities[to].degree++;
                to_adj[ins_pos].comm_id = p2;
                to_adj[ins_pos].dq = from_adj[j].dq - 2*to_a*communities[p2].a;
                if (to_adj[ins_pos].dq > max_dq) {
                    max_dq = to_adj[ins_pos].dq;
                    max_dq_idx = curr_i_idx;
                }
                /* Update (p2 -> from) */
                aggc_update_dq_p2(communities, dq_maxheap, p2, from, to,
                        to_adj[ins_pos].dq);
                j++;
                curr_i_idx++;
                continue;
            }
            
        }

        while (i < to_size) {
            p1 = to_adj[i].comm_id;
            if (p1 == from) {
                /* This community needs to be deleted */
                to_adj[i].comm_id = HIGH_VID;
                i++;
                continue;
            }
            to_adj[i].dq -= 2*from_a*communities[p1].a;
            if (to_adj[i].dq > max_dq) {
                max_dq = to_adj[i].dq;
                max_dq_idx = curr_i_idx;
            }
            /* Update dq value of commpair (p1 -> to) */
            aggc_update_dq_p1(communities, dq_maxheap, p1, to, to_adj[i].dq);
            i++;
            curr_i_idx++;
        }

        while (j < from_size) {
            p2 = from_adj[j].comm_id;
            if (p2 == to) {
                j++;
                continue;
            }
            if (communities[to].degree == communities[to].max_size) {
                aggc_adjcomm_resize(communities, to, from_size);
                to_adj = communities[to].adjcomm;
            } 
            ins_pos = communities[to].degree;
            communities[to].degree++;
            to_adj[ins_pos].comm_id = p2;
            to_adj[ins_pos].dq = from_adj[j].dq - 2*to_a*communities[p2].a;
            if (to_adj[ins_pos].dq > max_dq) {
                max_dq = to_adj[ins_pos].dq;
                max_dq_idx = curr_i_idx;
            }
            /* Update (p2 -> from) */
            aggc_update_dq_p2(communities, dq_maxheap, p2, from, to,
                    to_adj[ins_pos].dq);
            j++;
            curr_i_idx++;
        }

        assert(i == to_size);
        assert(j == from_size);

        /*
        for (i=0; i<communities[to].degree; i++) {
            fprintf(stderr, "[%d %ld] ", communities[to].adjcomm[i].comm_id,
                    communities[to].adjcomm[i].dq);
        }
        */
        qsort(communities[to].adjcomm, communities[to].degree, 
            sizeof(commpair_aggc_t), comp_comm);

        communities[to].a += communities[from].a;

        communities[to].degree -= 1;
        communities[to].comm_size += 1;
        
        communities[from].parent_id = to;
         
        if (communities[from].resized == 1) {
            free(communities[from].adjcomm);
            communities[from].resized = 0;
        }
       
        heap_pos = dq_index[from];
        /*
        if (max_dq_old != dq_comm_list[heap_pos].dq) {
            fprintf(stderr, "%lf %lf\n", max_dq_old, dq_comm_list[heap_pos].dq);
        }
        */
        aggc_maxheap_remove(dq_maxheap, from);

        communities[to].max_dq = max_dq;
        communities[to].max_dq_idx = max_dq_idx;

        heap_pos = dq_index[to];
        assert(heap_pos != -1);

        dq_comm_list[heap_pos].dq = max_dq;
        if (max_dq > max_dq_old)
            aggc_maxheap_sift_up(dq_maxheap, heap_pos);
        else
            aggc_maxheap_sift_down(dq_maxheap, heap_pos);
} 

/* Greedy community detection algorithms that optimize modularity. These are 
 * agglomerative approaches, i.e., we start by assuming a partitioning of
 * singleton communities, and then repeatedly merge communities with the 
 * goal of maximizing modularity. There are several heuristics for this, and we
 * use a similar representation of communities for all of them. */

void modularity_greedy_agglomerative(graph_t *g, char *alg_type, 
        attr_id_t *membership, attr_id_t *num_communities, double *modularity) {
    

    aggc_comm_t *communities; /* Size n to start off. The comm. adjacency lists
                                 can be resized when necessary. */ 
    aggc_maxheap_t* dq_maxheap; /* Max heap which stores community pairs that may 
                                  result in the max modularity change if merged */

    commpair_aggc_t *comm_memchunk;
    double mod_val;
    double mod_val_l;
    long num_uniq_edges;
    commpair_aggc_t *dq_comm_list;
    attr_id_t curr_pair;
    attr_id_t *dq_index;
    const attr_id_t* numEdges;
    const attr_id_t* endV;
    attr_id_t n, m;
    attr_id_t i, j, v;
    attr_id_t start_iter, end_iter, new_start_iter;
    double max_dq, new_dq, num_uniq_edges_2inv, num_uniq_edges_inv;
    attr_id_t max_dq_idx, uniq_adj_count, v_degree_i, degree_i; 
    commpair_aggc_t *comm_adj_ptr, *adj_buffer;
    attr_id_t no_of_joins, total_joins;
    double INFTY;
    attr_id_t num_comm_pairs;    /* No. of community pairs 
                                    is initially n, but keeps
                                    reducing. */
    attr_id_t parent_id, final_num_communities;

    INFTY = 100000.0;

    n = g->n;
    m = g->m;        /* we store each undirected edge twice, 
                     and so m = 2 * the actual no. of edges */
    numEdges = g->numEdges;
    endV = g->endV;
    mod_val = 0.0;

    /* Initialize */
    num_comm_pairs = 0;

    comm_memchunk = (commpair_aggc_t *) malloc(m * sizeof(commpair_aggc_t));
    adj_buffer = (commpair_aggc_t *) malloc(n * sizeof(commpair_aggc_t));
    communities = (aggc_comm_t *) malloc(n * sizeof(aggc_comm_t));
    curr_pair = 0;

    num_uniq_edges = 0;

    for (i=0; i<n; i++) {
        communities[i].parent_id = -1;
        communities[i].resized = 0;
        communities[i].comm_size = 1;
        start_iter = numEdges[i];
        end_iter   = numEdges[i+1];
        v_degree_i   = end_iter - start_iter;
        communities[i].max_size = v_degree_i;
        for (j=start_iter; j<end_iter; j++) {
            v = endV[j];
            comm_memchunk[j].comm_id = v;
        }
        qsort(comm_memchunk+start_iter, v_degree_i, sizeof(commpair_aggc_t), comp_comm);
        uniq_adj_count = 0;
        if (v_degree_i > 0) {
            if (comm_memchunk[start_iter].comm_id != i) {
                uniq_adj_count = 1;
                for (j=start_iter+1; j<end_iter; j++) {
                    if (comm_memchunk[j].comm_id != i) {
                        if (comm_memchunk[j].comm_id != 
                            comm_memchunk[start_iter+uniq_adj_count-1].comm_id) {

                            comm_memchunk[start_iter+uniq_adj_count++].comm_id 
                                = comm_memchunk[j].comm_id;
                        }
                    }
                }
            } else {
                uniq_adj_count = 0;
                for (j=start_iter+1; j<end_iter; j++) {
                    if (comm_memchunk[j].comm_id != i) {
                        comm_memchunk[start_iter+uniq_adj_count++].comm_id
                            = comm_memchunk[j].comm_id;
                        new_start_iter = j;
                        break;
                    }
                }
                if (j < end_iter) {
                    uniq_adj_count = 1;
                    for (j=new_start_iter+1; j<end_iter; j++) {
                        if (comm_memchunk[j].comm_id != i) {
                            if (comm_memchunk[j].comm_id != 
                                comm_memchunk[start_iter+uniq_adj_count-1].comm_id) {

                                comm_memchunk[start_iter+uniq_adj_count++].comm_id 
                                = comm_memchunk[j].comm_id;
                            }
                        }
                    }
                }
            }
        }
        communities[i].adjcomm = comm_memchunk+start_iter;
        communities[i].degree = uniq_adj_count;
        communities[i].a = uniq_adj_count;
        num_uniq_edges += uniq_adj_count;
    }
    num_uniq_edges_2inv = 1.0/(num_uniq_edges * num_uniq_edges);
    num_uniq_edges_inv  = 1.0/(num_uniq_edges);

    for (i=0; i<n; i++) {
        max_dq = -INFTY;
        max_dq_idx = -1;
        degree_i = communities[i].degree;
        comm_adj_ptr = communities[i].adjcomm;
        communities[i].a = communities[i].a * num_uniq_edges_inv;
        for (j=0; j<degree_i; j++) {
            v = comm_adj_ptr[j].comm_id;
            comm_adj_ptr[j].dq = 2 * (num_uniq_edges_inv - (degree_i *
                    communities[v].degree) * num_uniq_edges_2inv);
            if (comm_adj_ptr[j].dq > max_dq) {
                max_dq = comm_adj_ptr[j].dq;
                max_dq_idx = j;
            }
        }
        communities[i].max_dq = max_dq;
        communities[i].max_dq_idx = max_dq_idx;
    } 

    /* DEBUG */
    
    fprintf(stderr, "No. of edges after removing self loops"
           " and duplicates: %ld\n", num_uniq_edges/2); 
    /*
    for (i=0; i<n; i++) {
        fprintf(stderr, "Comm %d, max size %d, degree %d\n", i, 
                communities[i].max_size, communities[i].degree);
        fprintf(stderr, "max_dq %d, idx %d, parent id %d, a %ld, resized %d\n",
                communities[i].max_dq, 
                communities[i].max_dq_idx, communities[i].parent_id, 
                communities[i].a, communities[i].resized);
        for (j=0; j<communities[i].degree; j++) {
            fprintf(stderr, "[%d %ld] ", communities[i].adjcomm[j].comm_id, 
                    communities[i].adjcomm[j].dq);
        }
        fprintf(stderr, "\n\n");
    }
    */

    /* Compute initial modularity */
    mod_val_l = 0;
    mod_val = 0.0;
    for (i=0; i<n; i++) {
        mod_val -= communities[i].a * communities[i].a;
    }

    dq_maxheap = (aggc_maxheap_t *) malloc(sizeof(aggc_maxheap_t));
    assert(dq_maxheap != NULL);

    dq_maxheap->heap = (commpair_aggc_t *) malloc((n+1) * sizeof(commpair_aggc_t));
    assert(dq_maxheap->heap != NULL);
    dq_maxheap->index = (attr_id_t *) malloc(n * sizeof(attr_id_t));
    assert(dq_maxheap->index != NULL);

    dq_comm_list = dq_maxheap->heap;
    dq_index     = dq_maxheap->index;
    
    num_comm_pairs = 0;
    dq_maxheap->n = 0;
    
    /* heapify */
    for (i=0; i<n; i++) {
        if (communities[i].max_dq == -INFTY) {
            dq_index[i] = -1;
            continue;
        }
        dq_index[i] = dq_maxheap->n;
        max_dq = communities[i].max_dq;
        dq_comm_list[num_comm_pairs].comm_id = i;
        dq_comm_list[num_comm_pairs].dq = max_dq;
        aggc_maxheap_sift_up(dq_maxheap, dq_maxheap->n);
        dq_maxheap->n += 1;
        num_comm_pairs++;
     }
        
     /*
     fprintf(stderr, "No. of communities: %d\n", num_comm_pairs);
     */

     aggc_maxheap_check(dq_maxheap);

     /* DEBUG */
     /*   
     for (i=0; i<num_comm_pairs; i++) {
         fprintf(stderr, "%d -- %d\n", dq_maxheap->heap[i].dq, 
         dq_maxheap->heap[i].comm_id);
     }

     for (i=0; i<num_comm_pairs; i++) {
         fprintf(stderr, "%d, %d\n", i, dq_maxheap->heap[dq_index[i]].comm_id);
     }
     */

    total_joins = num_comm_pairs-1;

    no_of_joins = 0;
    /*fprintf(stderr, "Initial modularity: %lf\n", mod_val); */
    while (no_of_joins < total_joins) {

        aggc_merge_communities_cnm(communities, dq_maxheap,
                &new_dq, adj_buffer);
        
        aggc_maxheap_check(dq_maxheap);
        no_of_joins++;
        
        if (new_dq < 0) {
            break;
        }

        mod_val += new_dq;
        if ((no_of_joins % 10000) == 0)
            fprintf(stderr, "join: %d, mod %f\n", no_of_joins, mod_val);
    }

    /* get final community membership information */
    final_num_communities = 0;
    for (i=0; i<n; i++) {
        if (communities[i].parent_id == -1) {
            membership[i] = final_num_communities++;                        
        }
    }

    for (i=0; i<n; i++) {
        if (communities[i].parent_id != -1) {
            parent_id = communities[i].parent_id;
            while (communities[parent_id].parent_id != -1) {
                parent_id = communities[parent_id].parent_id;
            }
            membership[i] = membership[parent_id];
        }
    }

    *num_communities = final_num_communities;
    *modularity = mod_val;

    for (i=0; i<n; i++) {
        if (communities[i].resized == 1)
            free(communities[i].adjcomm);
    }

    free(adj_buffer);
    free(comm_memchunk);
    free(communities);
    free(dq_maxheap->heap);
    free(dq_maxheap->index);
    free(dq_maxheap);
}

