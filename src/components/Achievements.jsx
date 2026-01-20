import { useTracker } from '../context/TrackerContext';
import { Award, CheckCircle, Lock } from 'lucide-react';

const Achievements = () => {
    const { planData, completedTasks } = useTracker();

    // Extract milestones (Certs & Projects)
    const milestones = [];

    planData.phases.forEach(phase => {
        phase.months.forEach(month => {
            month.weeks.forEach(week => {
                week.tasks.forEach((task, idx) => {
                    const taskId = `${week.id}-${idx}`;
                    const isCompleted = completedTasks.includes(taskId);

                    if (task.toLowerCase().includes("certificate") || task.toLowerCase().includes("cert")) {
                        milestones.push({ type: 'cert', text: task, id: taskId, completed: isCompleted });
                    }
                    if (task.toLowerCase().includes("project") && !task.toLowerCase().includes("plan")) {
                        milestones.push({ type: 'project', text: task, id: taskId, completed: isCompleted });
                    }
                });
            });
        });
    });

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold">Achievements</h1>

            <div className="space-y-6">
                <div>
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <Award className="text-pink-500" /> Certifications
                    </h2>
                    <div className="grid grid-cols-1 gap-4">
                        {milestones.filter(m => m.type === 'cert').map((m, i) => (
                            <div key={i} className={`glass-panel p-4 flex items-center gap-4 ${m.completed ? 'border-primary/50 bg-primary/5' : 'opacity-60'}`}>
                                <div className={`p-3 rounded-full ${m.completed ? 'bg-primary text-black' : 'bg-gray-800 text-gray-500'}`}>
                                    {m.completed ? <CheckCircle size={20} /> : <Lock size={20} />}
                                </div>
                                <span className={`font-medium ${m.completed ? 'text-white' : 'text-gray-400'}`}>{m.text}</span>
                            </div>
                        ))}
                    </div>
                </div>

                <div>
                    <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <CheckCircle className="text-purple-500" /> Projects
                    </h2>
                    <div className="grid grid-cols-1 gap-4">
                        {milestones.filter(m => m.type === 'project').map((m, i) => (
                            <div key={i} className={`glass-panel p-4 flex items-center gap-4 ${m.completed ? 'border-primary/50 bg-primary/5' : 'opacity-60'}`}>
                                <div className={`p-3 rounded-full ${m.completed ? 'bg-primary text-black' : 'bg-gray-800 text-gray-500'}`}>
                                    {m.completed ? <CheckCircle size={20} /> : <Lock size={20} />}
                                </div>
                                <span className={`font-medium ${m.completed ? 'text-white' : 'text-gray-400'}`}>{m.text}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Achievements;
