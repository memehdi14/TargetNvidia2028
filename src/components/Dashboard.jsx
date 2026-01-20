import { useTracker } from '../context/TrackerContext';
import ProgressBar from './ProgressBar';
import { Award, Briefcase, Code, CheckCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
    const { planData, completedTasks, stats, progress, updateStat } = useTracker();
    const navigate = useNavigate();

    // Calculate Certs & Projects
    let completedCerts = 0;
    let completedProjects = 0;

    planData.phases.forEach(phase => {
        phase.months.forEach(month => {
            month.weeks.forEach(week => {
                week.tasks.forEach((task, idx) => {
                    const taskId = `${week.id}-${idx}`;
                    if (completedTasks.includes(taskId)) {
                        if (task.toLowerCase().includes("certificate") || task.toLowerCase().includes("cert")) {
                            completedCerts++;
                        }
                        if (task.toLowerCase().includes("project") && !task.toLowerCase().includes("plan")) {
                            completedProjects++;
                        }
                    }
                });
            });
        });
    });
    let activeWeek = null;
    let activeMonth = null;

    // Find the first week that has uncompleted tasks
    for (const phase of planData.phases) {
        if (activeWeek) break;
        for (const month of phase.months) {
            if (activeWeek) break;
            for (const week of month.weeks) {
                const iscomplete = week.tasks.every((_, idx) => completedTasks.includes(`${week.id}-${idx}`));
                if (!iscomplete) {
                    activeWeek = week;
                    activeMonth = month;
                    break;
                }
            }
        }
    }

    // Fallback if all done
    if (!activeWeek) {
        activeWeek = { title: "All Done!", focus: "You are ready for NVIDIA!", tasks: [] };
        activeMonth = { name: "Victory" };
    }

    const handleStatUpdate = (key, value) => {
        // Ensure value is a number
        const num = parseInt(value);
        if (!isNaN(num)) {
            updateStat(key, num); // Note: need to destructure updateStat from logic above or import
        }
    };

    const StatCard = ({ icon: Icon, label, value, total, color, isEditable, statKey }) => (
        <div className="glass-panel p-5 flex flex-col gap-3">
            <div className="flex justify-between items-start">
                <div className={`p-2 rounded-lg bg-[${color}20] text-[${color}]`}>
                    <Icon color={color} size={24} />
                </div>
                <div className="text-right">
                    {isEditable ? (
                        <div className="flex items-center gap-1 justify-end">
                            <input
                                type="number"
                                className="bg-transparent text-2xl font-bold w-16 text-right outline-none border-b border-white/10 focus:border-primary"
                                value={value}
                                onChange={(e) => updateStat(statKey, parseInt(e.target.value) || 0)}
                            />
                            <span className="text-muted text-sm leading-none">/{total}</span>
                        </div>
                    ) : (
                        <span className="text-2xl font-bold">{value}<span className="text-muted text-sm">/{total}</span></span>
                    )}
                </div>
            </div>
            <div>
                <h3 className="text-muted text-sm mb-2">{label}</h3>
                <ProgressBar progress={Math.min((value / total) * 100, 100)} color={color} height="6px" />
            </div>
        </div>
    );

    // Calculate Status
    const today = new Date();
    const targetDate = new Date('2027-07-01');
    const daysRemaining = Math.ceil((targetDate - today) / (1000 * 60 * 60 * 24));
    const isSaturday = today.getDay() === 6;

    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const formattedDate = today.toLocaleDateString('en-US', options);

    // Active Week Logic (from previous step)
    const remainingInWeek = activeWeek.tasks.length - activeWeek.tasks.filter((_, i) => completedTasks.includes(`${activeWeek.id}-${i}`)).length;

    return (
        <div className="space-y-8">
            {/* Hero Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Welcome & Time */}
                <div className="lg:col-span-2 glass-panel p-6 relative overflow-hidden flex flex-col justify-between min-h-[200px]">
                    <div className="absolute top-0 right-0 p-3 opacity-20 hover:opacity-100 transition-opacity">
                        <div className="text-sm font-mono text-primary animate-pulse">{daysRemaining} DAYS REMAINING</div>
                    </div>

                    <div>
                        <h2 className="text-muted uppercase tracking-widest text-sm font-semibold mb-1">{formattedDate}</h2>
                        <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
                            {isSaturday ? "It's Game Day." : "Keep the Streak."}
                        </h1>
                        <p className="text-gray-400 max-w-md">
                            {isSaturday
                                ? "Saturday is sacred. 10 hours of deep work. No excuses."
                                : "Weekdays are for prep. Squeeze in 1-2 hours of low-friction study."}
                        </p>
                    </div>

                    <div className="mt-6 flex items-center gap-4">
                        <div className="px-4 py-2 rounded-lg bg-primary/20 border border-primary/50 text-white font-mono text-sm">
                            Target: July 2027
                        </div>
                        {remainingInWeek > 0 && (
                            <div className="text-sm text-red-400 flex items-center gap-2 font-medium">
                                <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                                {remainingInWeek} Pending Tasks This Week
                            </div>
                        )}
                    </div>
                </div>

                {/* Current Focus Mini-Card */}
                <div className="glass-panel p-6 flex flex-col justify-center border-primary/30 relative">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary to-transparent" />
                    <h3 className="text-sm text-muted uppercase tracking-wider mb-3">CURRENT OBJECTIVE</h3>
                    <div className="text-2xl font-bold text-white mb-1 leading-tight">
                        {activeWeek.focus}
                    </div>
                    <p className="text-sm text-gray-400 mb-4">{activeMonth.name} • {activeWeek.title}</p>

                    <button
                        onClick={() => navigate('/plan')}
                        className="mt-auto w-full py-2 rounded-lg bg-white/5 hover:bg-primary hover:text-black transition-all font-medium text-sm border border-white/10"
                    >
                        View tasks ({remainingInWeek} left)
                    </button>
                </div>
            </div>

            {/* Main Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <StatCard
                    icon={Award}
                    label="Certifications"
                    value={completedCerts}
                    total={planData.stats.totalCerts}
                    color="#ec4899"
                />
                <StatCard
                    icon={Briefcase}
                    label="Projects"
                    value={completedProjects}
                    total={planData.stats.totalProjects}
                    color="#8b5cf6"
                />
                <StatCard
                    icon={Code}
                    label="LeetCode (Click to Update)"
                    value={stats.leetCodeSolved || 0}
                    total={planData.stats.totalLeetCode}
                    color="#76b900"
                    isEditable={true}
                    statKey="leetCodeSolved"
                />
            </div>

            {/* Main Progress */}
            <div className="glass-panel p-6 border-primary/20 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-10">
                    <Trophy size={100} /> {/* Need to import Trophy */}
                </div>
                <h2 className="text-xl font-bold mb-4">Total Progress</h2>
                <div className="flex items-end gap-2 mb-2">
                    <span className="text-4xl font-bold text-primary">{progress}%</span>
                    <span className="text-muted mb-1">completed</span>
                </div>
                <ProgressBar progress={progress} height="12px" />
            </div>

            {/* Current Focus */}
            <div className="glass-panel p-6">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-bold flex items-center gap-2">
                        <CheckCircle size={20} className="text-primary" />
                        Current Grind
                    </h2>
                    <button
                        onClick={() => navigate('/plan')}
                        className="text-sm text-primary hover:underline"
                    >
                        View Plan
                    </button>
                </div>

                <div className="bg-surface/50 p-4 rounded-xl border border-white/5">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-primary font-mono">{activeMonth.name} • {activeWeek.title}</span>
                    </div>
                    <h3 className="text-xl font-bold mb-2">{activeWeek.focus}</h3>
                    <p className="text-muted text-sm">{activeWeek.tasks.length} tasks remaining in this week</p>
                </div>
            </div>
        </div>
    );
};

// Re-importing Trophy specifically
import { Trophy } from 'lucide-react';

export default Dashboard;
