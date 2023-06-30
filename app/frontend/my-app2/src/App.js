import React, { useEffect, useState } from "react";
import { Stage, Layer, Group, Line, Circle, Image } from 'react-konva';

const AnnotationArea = () => {

  const [image, setImage] = useState(null);

  // Polygon management
  const [polygons, setPolygons] = useState({});
  const [polygonCounter, setPolygonCounter] = useState(0);
  const [currentPolygon, setCurrentPolygon] = useState([]);
  const currentPolygonRef = React.useRef(currentPolygon);

  const [previewLine, setPreviewLine] = useState(null);
  const isDrawing = React.useRef(false);

  // Hook for keeping track of lines
  useEffect(() => {
    currentPolygonRef.current = currentPolygon;
  }, [currentPolygon]);

  // Hook for registering keydown events -- happens only when component is mounted
  useEffect(() => {

    const image = new window.Image();
    image.src = 'https://konvajs.org/assets/yoda.jpg';
    image.onload = () => {
      setImage(image);
    };

    // Handling keydown events -- registering callback
    const handleKeyDown = (event) => {
      if (event.key === 'r') {
        resetCurrentPolygon();
      }
      else if (event.key === 'z' && event.ctrlKey) {
        undo();
      }
      else if (event.key == 'Escape') {
        stopDrawing();
      }
      else if (event.key == 's') {
        saveMask();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const handleMouseDown = (e) => {
    isDrawing.current = true;
    if (previewLine) {
      setCurrentPolygon([...currentPolygon, previewLine]);
      setPreviewLine(null);
    } else {
      const pos = e.target.getStage().getPointerPosition();
      setCurrentPolygon([...currentPolygon, { points: [pos.x, pos.y, pos.x, pos.y] }]);
    };
  }

  const handleMouseMove = (e) => {
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    if (!isDrawing.current || currentPolygon.length == 0) {
      return;
    }

    const lastLineEnd = currentPolygon[currentPolygon.length - 1].points.slice(2);

    // Computing current distance to first point
    const firstPoint = currentPolygon[0].points;
    const dx = firstPoint[0] - point.x;
    const dy = firstPoint[1] - point.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < 15) {
      const firstPoint = currentPolygon[0].points;
      setPreviewLine({ points: [...lastLineEnd, ...firstPoint] });
    } else {
      setPreviewLine({ points: [...lastLineEnd, point.x, point.y] });
    }

  };

  const handleMouseUp = () => {
    // isDrawing.current = false;
  };

  function stopDrawing() {
    saveMask();
    isDrawing.current = false;
    setPreviewLine(null);
  }

  function saveMask() {

    if (currentPolygonRef.current.length <= 2) {
      resetCurrentPolygon();
      return;
    }

    setPolygonCounter((prevCount) => {
      setPolygons((prevPolygons) => {
        return { ...prevPolygons, [prevCount]: currentPolygonRef.current.slice(0) };
      });
      return prevCount + 1;
    });
    setCurrentPolygon([]);
  }

  function resetCurrentPolygon() {
    setCurrentPolygon([]);
    setPreviewLine(null);
  }

  function reset() {
    setPolygons({});
    setCurrentPolygon([]);
    setPolygonCounter(0);
    setPreviewLine(null);
  }

  function undo() {
    setPreviewLine(null);
    setCurrentPolygon((lines) => lines.slice(0, -1));
  }

  return (
    <Stage
      width={window.innerWidth}
      height={window.innerHeight}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      <Layer>
        {image && (
          <Image
            image={image}
            x={0}
            y={0}
            width={window.innerWidth}
            height={window.innerHeight}
          />
        )}
        {Object.entries(polygons).map(([polygonId, polygon]) => {
          return <Polygon id={polygonId} lines={polygon} />
        })}
        <Polygon lines={currentPolygon} />
        {previewLine && (
          <Line
            points={previewLine.points}
            stroke="#df4b26"
            strokeWidth={5}
            tension={0.5}
            lineCap="round"
            lineJoin="round"
          />
        )}
      </Layer>
    </Stage>
  );
};

function Polygon({ id, lines }) {

  return (
    <Group>
      {lines.map((line, i) => (
        <>
          <Line
            key={i}
            points={line.points}
            stroke="#df4b26"
            strokeWidth={2}
            tension={0.2}
          />
          {
            <Circle
              x={line.points[2]}
              y={line.points[3]}
              radius={5}
              fill="black"
            />
          }
        </>
      ))}
    </Group >
  );
};

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      counter: 0,
      image: null,
    }
  }

  render() {
    return (
      <div className="App">
        <div className="App-header">
          <AnnotationArea />
        </div>
      </div>
    );
  }
}

export default App;
