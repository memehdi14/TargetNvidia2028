const ProgressBar = ({ progress, color = "var(--primary-color)", height = "8px" }) => {
    return (
        <div className="w-full bg-[#333] rounded-full overflow-hidden" style={{ height }}>
            <div
                className="h-full transition-all duration-1000 ease-out"
                style={{ width: `${progress}%`, backgroundColor: color }}
            />
        </div>
    );
};

export default ProgressBar;
