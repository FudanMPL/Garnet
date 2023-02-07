/*
 * Worker.h
 *
 */

#ifndef TOOLS_WORKER_H_
#define TOOLS_WORKER_H_

#include "WaitQueue.h"

template <class T>
class Worker
{
	pthread_t thread;
	WaitQueue<T*> input;
	WaitQueue<int> output;
	Timer timer, wall_timer;
	Timer request_timer, done_timer;
	int n_jobs;

	static void* run_thread(void* worker)
	{
		((Worker*)worker)->run();
		return 0;
	}

	void run()
	{
		T* job = 0;
		while (input.pop(job))
		{
#ifdef WORKER_TIMINGS
			request_timer.stop();
			TimeScope ts(timer), ts2(wall_timer);
#endif
			int res = job->run();
#ifdef WORKER_TIMINGS
			done_timer.start();
#endif
			output.push(res);
			n_jobs++;
		}
	}

public:
	Worker() : timer(CLOCK_THREAD_CPUTIME_ID)
	{
		pthread_create(&thread, 0, Worker::run_thread, this);
		n_jobs = 0;
	}

	~Worker()
	{
		input.stop();
		pthread_join(thread, 0);
#ifdef WORKER_TIMINGS
		cout << "Worker time: " << timer.elapsed() << endl;
		cout << "Worker wall time: " << wall_timer.elapsed() << endl;
		cout << "Request time: " << request_timer.elapsed() << endl;
		cout << "Done time: " << done_timer.elapsed() << endl;
		cout << "Run jobs: " << n_jobs << endl;
#endif
	}

	void request(T& job)
	{
#ifdef WORKER_TIMINGS
		request_timer.start();
#endif
		input.push(&job);
	}

	int done()
	{
		int res = 0;
		output.pop(res);
#ifdef WORKER_TIMINGS
		done_timer.stop();
#endif
		return res;
	}
};

#endif /* TOOLS_WORKER_H_ */
