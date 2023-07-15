import Button from '../Button/Button'
import './PopupMenu.css'


function PopupMenu({handleOptionClick, x, y}){
    console.log('Popup Menu props: ', { handleOptionClick, x, y })
    return(
        <div className="popup-menu" style={{ left: '${x}px', top: '${y}px'}}>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("rbc")}>Red Blood Cell</Button>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("wbc")}>White Blood Cell</Button>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("oof")}>Out of Focus</Button>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("agg")}>Aggregation</Button>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("plt")}>Platelet</Button>
            <Button className="popup-menu-button" onClick={() => handleOptionClick("n")}>None</Button>
        </div>) 
}

export{ PopupMenu}