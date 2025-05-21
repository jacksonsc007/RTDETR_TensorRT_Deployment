/*
The code is by courtesy of TinyChat. MIT, Han Song.
*/
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <iomanip>

#define PROFILE
#ifdef PROFILE
    #define STATS_START(x) Profiler::getInstance().start(x)
    #define STATS_END(x) Profiler::getInstance().stop(x)
    #define REPORT Profiler::getInstance().report_internal();
#else
    #define STATS_START(x) /* profiling disabled */
    #define STATS_END(x)   /* profiling disabled */
    #define REPORT         /* profiling disabled */
#endif

class Profiler {
   public:
    static Profiler& getInstance() {
        static Profiler instance;
        return instance;
    }

    void start(const std::string& section) { start_times[section] = std::chrono::high_resolution_clock::now(); }

    void start(const std::string& section, const long long section_flops) {
        start_times[section] = std::chrono::high_resolution_clock::now();
        if (flops.count(section) == 0)
            flops[section] = section_flops;
        else
            flops[section] += section_flops;
    }

    void reset() {
        start_times.clear();
        durations.clear();
        counts.clear();
        flops.clear();
    }

    void stop(const std::string& section) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_times[section]).count();
        durations[section] += duration;
        counts[section]++;
    }

    void report_internal() const {
        std::cout << std::left << std::setw(60) << "Section" 
                  << std::setw(30) << "Total time(ms)" 
                  << std::setw(30) << "Average time(ms)" 
                  << std::setw(30) << "Number of Images" 
                  << std::setw(30) << "OPS / FPS" 
                //   << std::setw(30) << "Mem BD GBS" 
                  << std::endl;

        for (const auto& entry : durations) {
            float total_time = (float)(entry.second) / 1000;
            float avg_time = total_time / (float) counts.at(entry.first);
            std::cout << std::left 
                    << std::setw(60) << entry.first 
                    << std::setw(30) << std::fixed << std::setprecision(2) << total_time
                    << std::setw(30) << std::fixed << std::setprecision(2) << avg_time
                    << std::setw(30) << counts.at(entry.first);
            float FPS = 1 / (avg_time / 1000);
            std::cout << std::setw(30) << std::fixed << std::setprecision(2) << FPS;
            std::cout << std::endl;
        }
    }
    void report() const {
#ifdef PROFILER
        report_internal();
#endif
    }

   private:
    Profiler() {}
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::map<std::string, long long> flops;
    std::map<std::string, long long> durations;
    std::map<std::string, int> counts;
};
