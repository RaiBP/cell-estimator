import Button from '../Button/Button'


function PopupMenu({handleOptionClick}){

    return(
        <div className="popup-menu">
            <Button onClick={() => handleOptionClick("rbc")}>RBC</Button>
            <Button onClick={() => handleOptionClick("wbc")}>WBC</Button>
            <Button onClick={() => handleOptionClick("oof")}>OOF</Button>
            <Button onClick={() => handleOptionClick("agg")}>AGG</Button>
            <Button onClick={() => handleOptionClick("plt")}>PLT</Button>
        </div>) 
}

export{ PopupMenu}