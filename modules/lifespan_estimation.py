from scipy.stats import norm

def lifespan_estimation(do_count, Y=500000, mean_life=10000000, std_life=5000000):
    p_x = norm.cdf(do_count, loc=mean_life, scale=std_life)
    p_xY = norm.cdf(do_count + Y, loc=mean_life, scale=std_life)
    P_T_gte_x = 1 - p_x
    P_T_gte_xY = 1 - p_xY
    cond_prob = P_T_gte_xY / P_T_gte_x

    return Y, cond_prob
