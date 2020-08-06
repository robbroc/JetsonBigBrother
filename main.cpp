#include "Process.h"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

void launch_all(void* data)
{

    Process proc("./face_detection/","./face_identification/");
    proc.run();
}


int main()
{	// tutto ciò risulta necessario per poter usare la GPU con sched_deadline
// e per poter gestire il thread che dialoga con la gpu ad alta priorità, evitando così che funga da collo di bottiglia
    sched_param fifo_params;
    fifo_params.sched_priority = 99;
    sched_setscheduler(pthread_self(), SCHED_FIFO,&fifo_params);
    int CudaDevice = cv::cuda::getDevice();
    cv::cuda::setDevice(CudaDevice);
    data_2_pass exec_param;
    exec_param.fun = &launch_all;
    exec_param.period = 50 * 1000 * 1000;
    exec_param.deadline = 50 * 1000 * 1000;
    exec_param.runtime = 49 * 1000 * 1000;

    pthread_t thread_master;

	pthread_create(&thread_master, nullptr, run_deadline , (void*)(&exec_param));
	pthread_join(thread_master, nullptr);
}



