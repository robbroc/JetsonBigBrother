#include "Process.h"

void launch_all(void* data)
{

    Process proc("./face_detection/","./face_identification/");
    proc.run();
}

int main()
{
    data_2_pass exec_param;
    exec_param.fun = &launch_all;
    exec_param.period = 40 * 1000 * 1000;
    exec_param.deadline = 40 * 1000 * 1000;
    exec_param.runtime = 35 * 1000 * 1000;

    pthread_t thread_master;

	pthread_create(&thread_master, NULL, run_deadline , (void*)(&exec_param));
	pthread_join(thread_master, nullptr);
}



