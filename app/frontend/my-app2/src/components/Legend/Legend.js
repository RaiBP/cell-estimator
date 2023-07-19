import './Legend.css';

function Legend() {
  return (
    <div className="legend-container">
      <div className="legend-item">
        <div className="legend-color-box" style={{ backgroundColor: '#ff0000' }}></div>
        Red Blood Cell
      </div>
      <div className="legend-item">
        <div className="legend-color-box" style={{ backgroundColor: '#ffff00' }}></div>
        Out of Focus
      </div>
      <div className="legend-item">
        <div className="legend-color-box" style={{ backgroundColor: '#00ff00' }}></div>
        Aggregate
      </div>
      <div className="legend-item">
        <div className="legend-color-box" style={{ backgroundColor: '#ffffff' }}></div>
        White Blood Cell
      </div>
      <div className="legend-item">
        <div className="legend-color-box" style={{ backgroundColor: '#0000ff' }}></div>
        Platelet
      </div>
    </div>
  );
}

export {Legend}