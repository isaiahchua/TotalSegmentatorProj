import time

def PrintTimeStats(start, end):
    start_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    end_s = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    time_taken = end - start
    if time_taken > 3600.:
        divisor = 3600.
        suffix = "hr"
    else:
        divisor = 60.
        suffix = "min"
    print("\n")
    print(f"Start Time: {start_s}, End Time: {end_s}, Total Time Taken: {(time_taken)/divisor:.3f} {suffix}")
