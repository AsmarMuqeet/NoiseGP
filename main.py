import GP as gp
import RP as rp
import os


if __name__ == '__main__':
    program_files = sorted(os.listdir("programs"))
    noise_files = os.listdir("models")
    #for program in program_files:
    for noise in noise_files:
        print("GP Running----------")
        gp.run_gp(program_files,noise,iterations=10)
        print("RP Running----------")
        rp.run_rp(program_files,noise,iterations=10)
        #break
        #break

    program_files = sorted(os.listdir("programs"))
    noise_files = os.listdir("models")
    gp.run_gp_real(program_files, "kyoto_0.pkl",iterations=10)
    rp.run_rp_real(program_files, "kyoto_0.pkl", iterations=10)