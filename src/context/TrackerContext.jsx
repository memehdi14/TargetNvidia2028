import { createContext, useContext, useState, useEffect } from 'react';
import { planData } from '../data/planData';

const TrackerContext = createContext();

export const TrackerProvider = ({ children }) => {
    const [completedTasks, setCompletedTasks] = useState(() => {
        const saved = localStorage.getItem('nvidia_tracker_tasks');
        return saved ? JSON.parse(saved) : [];
    });

    const [stats, setStats] = useState(() => {
        const saved = localStorage.getItem('nvidia_tracker_stats');
        return saved ? JSON.parse(saved) : { leetCodeSolved: 0 };
    });

    useEffect(() => {
        localStorage.setItem('nvidia_tracker_tasks', JSON.stringify(completedTasks));
    }, [completedTasks]);

    useEffect(() => {
        localStorage.setItem('nvidia_tracker_stats', JSON.stringify(stats));
    }, [stats]);

    const toggleTask = (taskId) => {
        setCompletedTasks(prev => {
            if (prev.includes(taskId)) {
                return prev.filter(id => id !== taskId);
            }
            return [...prev, taskId];
        });
    };

    const updateStat = (key, value) => {
        setStats(prev => ({ ...prev, [key]: value }));
    };

    const calculateProgress = () => {
        let totalTasks = 0;
        planData.phases.forEach(phase =>
            phase.months.forEach(month =>
                month.weeks.forEach(week =>
                    totalTasks += week.tasks.length
                )
            )
        );
        return Math.round((completedTasks.length / totalTasks) * 100);
    };

    return (
        <TrackerContext.Provider value={{
            planData,
            completedTasks,
            toggleTask,
            stats,
            updateStat,
            progress: calculateProgress()
        }}>
            {children}
        </TrackerContext.Provider>
    );
};

export const useTracker = () => useContext(TrackerContext);
