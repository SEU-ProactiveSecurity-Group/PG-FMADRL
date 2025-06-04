class Config:

    ALGORITHM = 'fed_multi_pg'  # 'dqn', 'double_dqn', 'actor_critic', 'wright_fisher', 'pg', 'fed_multi_pg'
    NUM_UAVS = 5
    COMM_RANGE = 500.0
    NUM_CHANNELS = 5
    GCS_RANGE = 1000.0
    FORMATION_DISTANCE = 400.0
    PATROL_RADIUS = 300.0
    PATROL_CENTER = [500.0, 500.0, 100.0]
    PATROL_HEIGHT = 100.0
    PATROL_SPEED = 15.0
    MAX_SPEED = 20.0

    EPISODE = 2000
    MAX_STEPS = 50
    MAX_CONSECUTIVE_ATTACK_STEPS = 10
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.001
    GAMMA = 0.99
    LR = 1e-3
    MEMORY_SIZE = 20000
    BATCH_SIZE = 128
     
    TARGET_UPDATE =  200
    FED_ROUND = 20
    LOCAL_FINETUNE_STEPS = 100

    ATTACK_TYPE = 'link'  # 'none', 'link', 'node'
    ATTACKER_TYPE = 'fixed'  # 'fixed', 'random', 'greedy'

    COMM_WEIGHT = 0.5
    FORMATION_WEIGHT = 0.5
    SPEED_PENALTY_WEIGHT = -0.5
    ATTACK_PENALTY_WEIGHT =  2
    COST_WEIGHT = -1