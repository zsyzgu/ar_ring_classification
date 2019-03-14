def get_file_info(file):
    status = file.split('.')[0].split('_')[0]
    name = file.split('.')[0].split('_')[1]
    ring = file.split('.')[0].split('_')[2]
    return status, name, ring

def qua_to_vec(q):
    return [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[0]*q[1] + q[2]*q[3]), 2*(0.5 - q[1]**2 - q[2]**2)]
