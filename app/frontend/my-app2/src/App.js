import React, { useEffect, useState } from "react";
import { Stage, Layer, Line, Circle, Image } from 'react-konva';

function ImageCanvas(props) {
  return (
    <div className="imagecanvas">
      <h1>{props.image}</h1>
    </div>
  );
}

const AnnotationArea = () => {

  const [image, setImage] = useState(null);
  const [lines, setLines] = useState([]);
  const [previewLine, setPreviewLine] = useState(null);
  const linesRef = React.useRef(lines);
  const isDrawing = React.useRef(false);

  // Hook for keeping track of lines
  useEffect(() => {
    linesRef.current = lines;
  }, [lines]);

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
        reset();
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
      setLines([...lines, previewLine]);
      setPreviewLine(null);
    } else {
      const pos = e.target.getStage().getPointerPosition();
      setLines([...lines, { points: [pos.x, pos.y, pos.x, pos.y] }]);
    };
    console.log(lines);
  }

  const handleMouseMove = (e) => {
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();

    if (!isDrawing.current) {
      return;
    }

    console.log(isDrawing.current);

    if (lines.length > 0) {
      const lastLineEnd = lines[lines.length - 1].points.slice(2);
      setPreviewLine({ points: [...lastLineEnd, point.x, point.y] });
    }
  };

  const handleMouseUp = () => {
    // isDrawing.current = false;
  };

  // Helpers
  function stopDrawing() {
    isDrawing.current = false;
    setPreviewLine(null);
  }

  function saveMask() {
    console.log(linesRef.current);
  }

  function reset() {
    setLines([]);
    setPreviewLine(null);
  }

  function undo() {
    setPreviewLine(null);
    setLines((lines) => lines.slice(0, -1));
  }


  return (
    <div>
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
    </div>
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

  incrementCounter() {
    this.setState({
      counter: this.state.counter + 1,
    });
    console.log(this.state.counter)
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
