#include "thread_utility.h"


 int sched_setattr_new(pid_t pid,
		  const struct sched_attr *attr,
		  unsigned int flags)
 {
	return syscall(__NR_sched_setattr, pid, attr, flags);
 }

 int sched_getattr_new(pid_t pid,
		  struct sched_attr *attr,
		  unsigned int size,
		  unsigned int flags)
 {
	return syscall(__NR_sched_getattr, pid, attr, size, flags);
 }

 void *run_deadline(void *data)
 {
    int ret;
	unsigned int flags = 0;
	struct data_2_pass* dati = (data_2_pass*) data;
	struct sched_attr attr;

	attr.size = sizeof(attr);
	attr.sched_flags = 0;
	attr.sched_nice = 0;
	attr.sched_priority = 0;

	/* This creates a 10ms/30ms reservation */
	attr.sched_policy = SCHED_DEADLINE;
	attr.sched_runtime = dati->runtime;
	attr.sched_period = dati->period;
	attr.sched_deadline = dati->deadline;

    ret = sched_setattr_new(0, &attr, flags);
	if (ret < 0) {
		perror("sched_setattr");
		exit(-1);
	}

	// eseguo la funzione
	(*(dati->fun))();


	return nullptr;
 }
