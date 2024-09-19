from sa_npu import SA_Accelerator
import argparse
import logging

logger = logging.getLogger()


def simulation(args):
    if args.debug:
        logger.setLevel(logging.DEBUG) 
    else:
        logger.setLevel(logging.INFO)

    if args.model == "resnet18":
    elif args.model == "resnet50":
    elif args.model == "mobilenet":
    elif args.model == "vit":
    else:
        raise NotImplementedError("Not supported model.")

    if args.sa_size == "small":
        block_size = 4
        sa_size = 2
        bram_size = 1024 * 2
        mux_size = 4
    elif args.sa_size == "base":
        block_size = 4
        sa_size = 4
        bram_size = 1024 * 4
        mux_size = 4
    elif args.sa_size == "big":
        block_size = 4
        sa_size = 8
        bram_size = 1024 * 8
        mux_size = 8      
    else:
        raise NotImplementedError("Not supported version.")

    design = SA_Accelerator(sa_size, sa_size, block_size, args.sparse, mux_size)
    design.matmul()

    network_run_cost = num_layer * design.run_cycles
    ms_per_clock = (1/args.frequency/1000) / args.efficiency
    print ("The overall latecy is:", network_run_cost*ms_per_clock) 
    logging.info("####################Finish######################")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="resnet50", type=str, help="model to run")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sa", default="base", type=str, help="Size of Systolic Array")
    parser.add_argument("--frequency", default=200, type=int, help="The frequency of the design")
    parser.add_argument("--efficiency", default=0.85, type=float, help="The hardware implementation efficiency")

    args = parser.parse_args()

    simulation(args)
